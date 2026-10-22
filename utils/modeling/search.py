"""Model-specific feature voting and Bayesian model selection for xG.

Expected workflow:
    1. Split raw X and y into development and untouched test sets in a notebook.
    2. Call fit_model_searches(X_development, y_development).
    3. Compare development CV results.
    4. Finalize the model choice without consulting the test set.
    5. Call evaluate_searches(..., X_test, y_test) once.

Every searched Pipeline contains preprocessing, voting feature selection, and
the classifier. Consequently, all learned preprocessing and feature decisions
are fitted only on the training portion of each BayesSearchCV fold.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install scikit-optimize with: pip install scikit-optimize"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install XGBoost with: pip install xgboost") from exc


RANDOM_STATE = 21
N_JOBS = -1
DEFAULT_SCORING = "neg_log_loss"

LOGGER = logging.getLogger("xg_model_selection")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False



@dataclass(frozen=True)
class ModelSpecification:
    """One estimator, its selector estimators, preprocessing, and search space."""

    estimator: BaseEstimator
    search_space: dict[str, Any]
    preprocessing: str
    rfe_estimator: BaseEstimator
    sequential_estimator: BaseEstimator


@dataclass(frozen=True)
class ModelSearchResult:
    """Outputs from one completed development-set search."""

    model_name: str
    search: BayesSearchCV
    selected_features: list[str]
    selection_report: pd.DataFrame

    @property
    def estimator(self) -> Pipeline:
        return self.search.best_estimator_

    @property
    def best_cv_log_loss(self) -> float:
        return -float(self.search.best_score_)


def make_stratified_cv(
    n_splits: int = 5,
    *,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _validate_xy(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indexes.")
    unsupported = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns
    if len(unsupported):
        raise ValueError(
            "Encode categorical columns before model selection: "
            f"{unsupported.tolist()}."
        )
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("y must be a complete binary target.")
    if X.shape[1] < 2:
        raise ValueError("At least two predictor columns are required.")


def _column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    booleans = X.select_dtypes(include="bool").columns.tolist()
    missing_booleans = [column for column in booleans if X[column].isna().any()]
    if missing_booleans:
        raise ValueError(
            "Boolean predictors must not contain missing values: "
            f"{missing_booleans}."
        )
    numeric = X.columns.difference(booleans, sort=False).tolist()
    return booleans, numeric


def make_linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute, add missing indicators, and scale numeric predictors."""
    booleans, numeric = _column_groups(X)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("boolean", "passthrough", booleans),
            ("numeric", numeric_pipeline, numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_native_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Preserve NaNs for estimators that learn missing-value branches."""
    booleans, numeric = _column_groups(X)
    return ColumnTransformer(
        [
            ("boolean", "passthrough", booleans),
            ("numeric", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_imputed_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Median-impute for estimators such as GradientBoostingClassifier."""
    booleans, numeric = _column_groups(X)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )
    return ColumnTransformer(
        [
            ("boolean", "passthrough", booleans),
            ("numeric", numeric_pipeline, numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _binary_complete_columns(X: pd.DataFrame) -> list[str]:
    """Identify unscaled, complete 0/1 columns eligible for chi-square."""
    eligible = []
    for column in X.columns:
        values = pd.Series(X[column]).dropna().unique()
        if len(values) == 2 and set(values).issubset({0, 1, False, True}):
            eligible.append(column)
    return eligible


class VotingFeatureSelector(TransformerMixin, BaseEstimator):
    """Vote across RFECV, forward SFS, backward SFS, and chi-square.

    The three wrapper methods vote for every predictor. Chi-square is used only
    for complete binary predictors. A binary predictor one vote below the
    threshold is rescued when its chi-square association is significant,
    preserving the decision rule from the original feature-selection module.
    """

    def __init__(
        self,
        rfe_estimator: BaseEstimator,
        sequential_estimator: BaseEstimator,
        *,
        min_votes: int = 2,
        alpha: float = 0.05,
        cv: int = 3,
        scoring: str = DEFAULT_SCORING,
        rfe_step: int | float = 0.20,
        min_features_to_select: int = 1,
        sfs_n_features: int | float | str = 0.5,
        n_jobs: int = 1,
        label: str = "model",
        verbose: int = 1,
    ) -> None:
        self.rfe_estimator = rfe_estimator
        self.sequential_estimator = sequential_estimator
        self.min_votes = min_votes
        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.rfe_step = rfe_step
        self.min_features_to_select = min_features_to_select
        self.sfs_n_features = sfs_n_features
        self.n_jobs = n_jobs
        self.label = label
        self.verbose = verbose

    def _log(self, message: str, *args: Any) -> None:
        if self.verbose:
            LOGGER.info("[%s selector] " + message, self.label, *args)

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        columns = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=columns)

    def fit(self, X: Any, y: Any) -> "VotingFeatureSelector":
        selector_started = perf_counter()
        frame = self._frame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]

        if not 1 <= self.min_votes <= 3:
            raise ValueError("min_votes must be between 1 and 3.")

        selector_cv = make_stratified_cv(
            self.cv,
            random_state=RANDOM_STATE + 100,
        )
        self._log(
            "starting with %d rows and %d features",
            frame.shape[0],
            frame.shape[1],
        )

        stage_started = perf_counter()
        self._log("running RFECV")
        rfecv = RFECV(
            estimator=clone(self.rfe_estimator),
            step=self.rfe_step,
            min_features_to_select=min(
                self.min_features_to_select,
                frame.shape[1],
            ),
            scoring=self.scoring,
            cv=selector_cv,
            n_jobs=self.n_jobs,
        ).fit(frame, y)
        self._log(
            "RFECV finished in %.1fs; selected %d features",
            perf_counter() - stage_started,
            int(rfecv.get_support().sum()),
        )

        stage_started = perf_counter()
        self._log("running forward sequential selection")
        forward = SequentialFeatureSelector(
            estimator=clone(self.sequential_estimator),
            n_features_to_select=self.sfs_n_features,
            direction="forward",
            scoring=self.scoring,
            cv=selector_cv,
            n_jobs=self.n_jobs,
        ).fit(frame, y)
        self._log(
            "forward selection finished in %.1fs; selected %d features",
            perf_counter() - stage_started,
            int(forward.get_support().sum()),
        )

        stage_started = perf_counter()
        self._log("running backward sequential selection")
        backward = SequentialFeatureSelector(
            estimator=clone(self.sequential_estimator),
            n_features_to_select=self.sfs_n_features,
            direction="backward",
            scoring=self.scoring,
            cv=selector_cv,
            n_jobs=self.n_jobs,
        ).fit(frame, y)
        self._log(
            "backward selection finished in %.1fs; selected %d features",
            perf_counter() - stage_started,
            int(backward.get_support().sum()),
        )

        votes = pd.DataFrame(
            {
                "RFECV": rfecv.get_support(),
                "Forward SFS": forward.get_support(),
                "Backward SFS": backward.get_support(),
            },
            index=frame.columns,
            dtype=bool,
        )
        vote_count = votes.sum(axis=1)

        chi_square = pd.Series(
            False,
            index=frame.columns,
            name="Chi-square",
            dtype=bool,
        )
        chi_square_pvalue = pd.Series(
            np.nan,
            index=frame.columns,
            name="Chi-square p-value",
        )
        self._log("running chi-square tests on complete binary features")
        for column in _binary_complete_columns(frame):
            observed = pd.crosstab(pd.Series(y, index=frame.index), frame[column])
            if observed.shape == (2, 2):
                pvalue = stats.chi2_contingency(observed).pvalue
                chi_square_pvalue.loc[column] = pvalue
                chi_square.loc[column] = pvalue < self.alpha

        selected = (
            vote_count.ge(self.min_votes)
            | (
                vote_count.eq(self.min_votes - 1)
                & chi_square
            )
        )

        # Defensive fallback: a valid pipeline must always retain a predictor.
        if not selected.any():
            selected.loc[vote_count == vote_count.max()] = True

        self.support_ = selected.to_numpy(dtype=bool)
        self.selection_report_ = votes.assign(
            **{
                "Chi-square": chi_square,
                "Chi-square p-value": chi_square_pvalue,
                "Vote count": vote_count,
                "Selected": selected,
            }
        )
        self._log(
            "finished in %.1fs; retained %d of %d features",
            perf_counter() - selector_started,
            int(self.support_.sum()),
            frame.shape[1],
        )
        return self

    def transform(self, X: Any) -> Any:
        check_is_fitted(self, "support_")
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.get_feature_names_out()]
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "support_")
        names = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )
        return names[self.support_]

    def get_selection_report(self) -> pd.DataFrame:
        check_is_fitted(self, "selection_report_")
        return self.selection_report_.copy()


def _fast_logistic(penalty: str = "l2") -> LogisticRegression:
    kwargs: dict[str, Any] = {
        "penalty": penalty,
        "solver": "saga",
        "C": 1.0,
        "max_iter": 3000,
        "random_state": RANDOM_STATE,
    }
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = 0.5
    return LogisticRegression(**kwargs)


def _fast_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _fast_extra_trees() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _fast_xgboost() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_model_registry() -> dict[str, ModelSpecification]:
    """Return all supported xG model specifications."""
    decision_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    gradient_selector = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )
    hist_sequential = HistGradientBoostingClassifier(
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )

    return {
        "logistic_l2": ModelSpecification(
            estimator=_fast_logistic("l2"),
            search_space={
                "model__C": Real(1e-3, 1e2, prior="log-uniform"),
            },
            preprocessing="linear",
            rfe_estimator=_fast_logistic("l2"),
            sequential_estimator=_fast_logistic("l2"),
        ),
        "logistic_l1": ModelSpecification(
            estimator=_fast_logistic("l1"),
            search_space={
                "model__C": Real(1e-3, 1e2, prior="log-uniform"),
            },
            preprocessing="linear",
            rfe_estimator=_fast_logistic("l1"),
            sequential_estimator=_fast_logistic("l1"),
        ),
        "logistic_elasticnet": ModelSpecification(
            estimator=_fast_logistic("elasticnet"),
            search_space={
                "model__C": Real(1e-3, 1e2, prior="log-uniform"),
                "model__l1_ratio": Real(0.05, 0.95),
            },
            preprocessing="linear",
            rfe_estimator=_fast_logistic("elasticnet"),
            sequential_estimator=_fast_logistic("elasticnet"),
        ),
        "decision_tree": ModelSpecification(
            estimator=clone(decision_tree),
            search_space={
                "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
                "model__max_depth": Integer(2, 12),
                "model__min_samples_split": Integer(2, 40),
                "model__min_samples_leaf": Integer(2, 30),
                "model__max_features": Categorical([None, "sqrt", "log2"]),
                "model__ccp_alpha": Real(1e-8, 0.05, prior="log-uniform"),
            },
            preprocessing="native_tree",
            rfe_estimator=clone(decision_tree),
            sequential_estimator=clone(decision_tree),
        ),
        "random_forest": ModelSpecification(
            estimator=_fast_random_forest(),
            search_space={
                "model__n_estimators": Integer(200, 900),
                "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
                "model__max_depth": Integer(3, 14),
                "model__min_samples_split": Integer(2, 30),
                "model__min_samples_leaf": Integer(2, 25),
                "model__max_features": Categorical(["sqrt", "log2", None]),
                "model__max_samples": Real(0.60, 1.0),
            },
            preprocessing="native_tree",
            rfe_estimator=_fast_random_forest(),
            sequential_estimator=_fast_random_forest(),
        ),
        "extra_trees": ModelSpecification(
            estimator=_fast_extra_trees(),
            search_space={
                "model__n_estimators": Integer(200, 900),
                "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
                "model__max_depth": Integer(3, 14),
                "model__min_samples_split": Integer(2, 30),
                "model__min_samples_leaf": Integer(2, 25),
                "model__max_features": Categorical(["sqrt", "log2", None]),
            },
            preprocessing="native_tree",
            rfe_estimator=_fast_extra_trees(),
            sequential_estimator=_fast_extra_trees(),
        ),
        "gradient_boosting": ModelSpecification(
            estimator=clone(gradient_selector),
            search_space={
                "model__n_estimators": Integer(100, 700),
                "model__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
                "model__max_depth": Integer(1, 5),
                "model__min_samples_split": Integer(2, 30),
                "model__min_samples_leaf": Integer(2, 25),
                "model__subsample": Real(0.60, 1.0),
                "model__max_features": Categorical([None, "sqrt", "log2"]),
            },
            preprocessing="imputed_tree",
            rfe_estimator=clone(gradient_selector),
            sequential_estimator=clone(gradient_selector),
        ),
        "hist_gradient_boosting": ModelSpecification(
            estimator=clone(hist_sequential),
            search_space={
                "model__max_iter": Integer(100, 700),
                "model__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
                "model__max_leaf_nodes": Integer(7, 63),
                "model__max_depth": Integer(2, 10),
                "model__min_samples_leaf": Integer(5, 40),
                "model__l2_regularization": Real(
                    1e-8, 20.0, prior="log-uniform"
                ),
                "model__max_bins": Integer(64, 255),
            },
            preprocessing="native_tree",
            # HistGradientBoosting has no coef_ or feature_importances_, so
            # Extra Trees supplies the tree-family RFECV ranking.
            rfe_estimator=_fast_extra_trees(),
            sequential_estimator=clone(hist_sequential),
        ),
        "xgboost": ModelSpecification(
            estimator=_fast_xgboost(),
            search_space={
                "model__n_estimators": Integer(150, 900),
                "model__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
                "model__max_depth": Integer(2, 6),
                "model__min_child_weight": Integer(1, 20),
                "model__subsample": Real(0.60, 1.0),
                "model__colsample_bytree": Real(0.60, 1.0),
                "model__gamma": Real(1e-8, 2.0, prior="log-uniform"),
                "model__reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
                "model__reg_lambda": Real(1e-3, 30.0, prior="log-uniform"),
            },
            preprocessing="native_tree",
            rfe_estimator=_fast_xgboost(),
            sequential_estimator=_fast_xgboost(),
        ),
    }


def _make_preprocessor(kind: str, X: pd.DataFrame) -> ColumnTransformer:
    if kind == "linear":
        return make_linear_preprocessor(X)
    if kind == "native_tree":
        return make_native_tree_preprocessor(X)
    if kind == "imputed_tree":
        return make_imputed_tree_preprocessor(X)
    raise ValueError(f"Unknown preprocessing kind: {kind}.")


def make_model_search(
    model_name: str,
    X: pd.DataFrame,
    *,
    n_iter: int = 20,
    bayes_cv_splits: int = 5,
    selector_cv_splits: int = 3,
    min_votes: int = 2,
    alpha: float = 0.05,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = N_JOBS,
    cache_dir: str | Path | None = None,
    verbose: int = 1,
) -> BayesSearchCV:
    """Construct one complete preprocessing-selection-model Bayesian search."""
    registry = make_model_registry()
    if model_name not in registry:
        raise KeyError(
            f"Unknown model {model_name!r}. Available: {list(registry)}"
        )
    specification = registry[model_name]

    selector = VotingFeatureSelector(
        rfe_estimator=specification.rfe_estimator,
        sequential_estimator=specification.sequential_estimator,
        min_votes=min_votes,
        alpha=alpha,
        cv=selector_cv_splits,
        scoring=scoring,
        rfe_step=0.20,
        min_features_to_select=1,
        sfs_n_features=0.5,
        n_jobs=1,
        label=model_name,
        verbose=verbose,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", _make_preprocessor(specification.preprocessing, X)),
            ("selector", selector),
            ("model", clone(specification.estimator)),
        ],
        memory=cache_dir,
    )
    return BayesSearchCV(
        estimator=pipeline,
        search_spaces=specification.search_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=make_stratified_cv(
            bayes_cv_splits,
            random_state=RANDOM_STATE,
        ),
        refit=True,
        return_train_score=False,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score="raise",
    )


def available_models() -> list[str]:
    """Return valid names for the models argument."""
    return list(make_model_registry())


def fit_model_searches(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    *,
    models: list[str] | None = None,
    n_iter: int = 20,
    bayes_cv_splits: int = 5,
    selector_cv_splits: int = 3,
    min_votes: int = 2,
    alpha: float = 0.05,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = N_JOBS,
    verbose: int = 1,
    cache_dir: str | Path | None = None,
) -> dict[str, ModelSearchResult]:
    """Tune requested models using only the notebook's development partition."""
    _validate_xy(X_development, y_development)
    requested = available_models() if models is None else models
    unknown = sorted(set(requested).difference(available_models()))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}.")

    results: dict[str, ModelSearchResult] = {}
    total_models = len(requested)
    overall_started = perf_counter()
    if verbose:
        LOGGER.info(
            "Starting model selection for %d models; %d Bayesian candidates "
            "per model and %d CV folds",
            total_models,
            n_iter,
            bayes_cv_splits,
        )

    for model_number, model_name in enumerate(requested, start=1):
        model_started = perf_counter()
        if verbose:
            LOGGER.info(
                "[%d/%d] Starting %s",
                model_number,
                total_models,
                model_name,
            )
        search = make_model_search(
            model_name,
            X_development,
            n_iter=n_iter,
            bayes_cv_splits=bayes_cv_splits,
            selector_cv_splits=selector_cv_splits,
            min_votes=min_votes,
            alpha=alpha,
            scoring=scoring,
            n_jobs=n_jobs,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        search.fit(X_development, y_development)
        selector = search.best_estimator_.named_steps["selector"]
        results[model_name] = ModelSearchResult(
            model_name=model_name,
            search=search,
            selected_features=selector.get_feature_names_out().tolist(),
            selection_report=selector.get_selection_report(),
        )
        if verbose:
            LOGGER.info(
                "[%d/%d] Finished %s in %.1fs | CV log loss %.6f | "
                "%d selected features",
                model_number,
                total_models,
                model_name,
                perf_counter() - model_started,
                results[model_name].best_cv_log_loss,
                len(results[model_name].selected_features),
            )

    if verbose:
        LOGGER.info(
            "All model searches finished in %.1fs",
            perf_counter() - overall_started,
        )
    return results


def compare_searches(
    results: dict[str, ModelSearchResult],
) -> pd.DataFrame:
    """Compare development CV scores; lower cv_log_loss is better."""
    rows = [
        {
            "model": name,
            "cv_log_loss": result.best_cv_log_loss,
            "n_selected_features": len(result.selected_features),
            "selected_features": result.selected_features,
            "best_parameters": result.search.best_params_,
        }
        for name, result in results.items()
    ]
    return pd.DataFrame(rows).sort_values("cv_log_loss", ignore_index=True)


def selection_reports(
    results: dict[str, ModelSearchResult],
) -> pd.DataFrame:
    """Combine final development-data voting reports for all searched models."""
    reports = []
    for model_name, result in results.items():
        report = result.selection_report.copy()
        report.insert(0, "Feature", report.index)
        report.insert(0, "Model", model_name)
        reports.append(report.reset_index(drop=True))
    return pd.concat(reports, ignore_index=True)


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.Series:
    """Calculate xG metrics from untouched out-of-sample probabilities."""
    actual = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    return pd.Series(
        {
            "log_loss": log_loss(actual, probabilities),
            "brier_score": brier_score_loss(actual, probabilities),
            "roc_auc": roc_auc_score(actual, probabilities),
            "average_precision": average_precision_score(actual, probabilities),
            "mean_xg": probabilities.mean(),
            "actual_goal_rate": actual.mean(),
            "expected_goals": probabilities.sum(),
            "actual_goals": actual.sum(),
        }
    )


def evaluate_searches(
    results: dict[str, ModelSearchResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate finalized pipelines once on the notebook's untouched test set."""
    requested = list(results) if models is None else models
    evaluations = {}
    for model_name in requested:
        probabilities = results[model_name].estimator.predict_proba(X_test)[:, 1]
        evaluations[model_name] = evaluate_probabilities(
            y_test,
            probabilities,
        )
    return pd.DataFrame(evaluations).T.sort_values("log_loss")


def clone_best_estimator(result: ModelSearchResult) -> Pipeline:
    """Return an unfitted copy of one winning end-to-end pipeline."""
    return clone(result.estimator)
