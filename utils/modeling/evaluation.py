from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def calculate_metrics(y_true, probabilities):
    return {
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
    }


def paired_bootstrap(
    y_true,
    predictions,
    *,
    reference_model="hist_gradient_boosting",
    n_bootstrap=5000,
    confidence_level=0.95,
    random_state=21,
):
    y_true = np.asarray(y_true)
    predictions = {
        name: np.asarray(probability)
        for name, probability in predictions.items()
    }

    if reference_model not in predictions:
        raise ValueError(
            f"{reference_model!r} is not in predictions."
        )

    rng = np.random.default_rng(random_state)

    # Stratified resampling preserves the number of goals and misses.
    goal_indices = np.flatnonzero(y_true == 1)
    miss_indices = np.flatnonzero(y_true == 0)

    metric_names = [
        "log_loss",
        "brier_score",
        "roc_auc",
        "average_precision",
    ]

    bootstrap_scores = {
        model: {
            metric: []
            for metric in metric_names
        }
        for model in predictions
    }

    for _ in range(n_bootstrap):
        sampled_goals = rng.choice(
            goal_indices,
            size=len(goal_indices),
            replace=True,
        )

        sampled_misses = rng.choice(
            miss_indices,
            size=len(miss_indices),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [sampled_goals, sampled_misses]
        )

        sampled_y = y_true[sampled_indices]

        for model_name, probability in predictions.items():
            sampled_probability = probability[
                sampled_indices
            ]

            scores = calculate_metrics(
                sampled_y,
                sampled_probability,
            )

            for metric, value in scores.items():
                bootstrap_scores[
                    model_name
                ][metric].append(value)

    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    estimates = []
    comparisons = []

    point_scores = {
        model: calculate_metrics(
            y_true,
            probability,
        )
        for model, probability in predictions.items()
    }

    # Confidence interval for each model's metric.
    for model_name in predictions:
        for metric in metric_names:
            samples = np.asarray(
                bootstrap_scores[model_name][metric]
            )

            estimates.append({
                "model": model_name,
                "metric": metric,
                "estimate": point_scores[
                    model_name
                ][metric],
                "ci_lower": np.percentile(
                    samples,
                    lower_percentile,
                ),
                "ci_upper": np.percentile(
                    samples,
                    upper_percentile,
                ),
                "bootstrap_std": samples.std(ddof=1),
            })

    # Paired difference: model minus reference model.
    for model_name in predictions:
        if model_name == reference_model:
            continue

        for metric in metric_names:
            model_samples = np.asarray(
                bootstrap_scores[model_name][metric]
            )

            reference_samples = np.asarray(
                bootstrap_scores[
                    reference_model
                ][metric]
            )

            differences = (
                model_samples - reference_samples
            )

            point_difference = (
                point_scores[model_name][metric]
                - point_scores[reference_model][metric]
            )

            lower = np.percentile(
                differences,
                lower_percentile,
            )

            upper = np.percentile(
                differences,
                upper_percentile,
            )

            comparisons.append({
                "model": model_name,
                "reference": reference_model,
                "metric": metric,
                "difference": point_difference,
                "ci_lower": lower,
                "ci_upper": upper,
                "includes_zero": lower <= 0 <= upper,
            })

    return (
        pd.DataFrame(estimates),
        pd.DataFrame(comparisons),
    )


def calibration_statistics(
    y_true,
    predictions,
    n_bins=8,
):
    y_true = np.asarray(y_true)
    rows = []

    for model_name, probability in predictions.items():
        probability = np.asarray(probability)

        clipped = np.clip(
            probability,
            1e-6,
            1 - 1e-6,
        )

        predicted_logit = np.log(
            clipped / (1 - clipped)
        ).reshape(-1, 1)

        calibration_model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=5000,
        )

        calibration_model.fit(
            predicted_logit,
            y_true,
        )

        calibration_intercept = (
            calibration_model.intercept_[0]
        )

        calibration_slope = (
            calibration_model.coef_[0, 0]
        )

        table = pd.DataFrame({
            "actual": y_true,
            "probability": probability,
        })

        table["bin"] = pd.qcut(
            table["probability"],
            q=n_bins,
            duplicates="drop",
        )

        binned = (
            table
            .groupby("bin", observed=True)
            .agg(
                count=("actual", "size"),
                observed_rate=("actual", "mean"),
                mean_probability=("probability", "mean"),
            )
        )

        weights = binned["count"] / len(table)

        ece = (
            weights
            * (
                binned["observed_rate"]
                - binned["mean_probability"]
            ).abs()
        ).sum()

        rows.append({
            "model": model_name,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
            "ece": ece,
            "mean_xg": probability.mean(),
            "actual_goal_rate": y_true.mean(),
        })

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("ece")
    )


def benchmark_inference(
    results,
    X,
    models,
    n_repeats=100,
):
    rows = []

    for model_name in models:
        estimator = results[model_name].estimator

        # Warm-up call
        estimator.predict_proba(X)

        durations = []

        for _ in range(n_repeats):
            started = perf_counter()
            estimator.predict_proba(X)
            durations.append(
                perf_counter() - started
            )

        rows.append({
            "model": model_name,
            "median_seconds": np.median(durations),
            "mean_seconds": np.mean(durations),
            "std_seconds": np.std(
                durations,
                ddof=1,
            ),
            "shots_per_second": (
                len(X) / np.median(durations)
            ),
        })

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("median_seconds")
    )
