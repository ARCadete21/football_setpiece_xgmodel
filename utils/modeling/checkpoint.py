from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
import gzip
import os
import pickle
import platform
import tempfile

import cloudpickle
import numpy as np
import pandas as pd

from .search import ModelSearchResult, evaluate_searches


def _package_version(package: str) -> str | None:
    """Return an installed package version without failing if unavailable."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def save_model_checkpoint(
    results: dict[str, ModelSearchResult],
    path: str | Path,
    *,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save fitted searches and optional evaluation data.

    The complete ModelSearchResult objects are preserved, including fitted
    preprocessing, feature selectors, estimators, Bayesian-search results,
    selected features, and selection reports.

    The file is written atomically so an interrupted write does not leave a
    partially written checkpoint.
    """
    if not results:
        raise ValueError("results cannot be empty.")

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "format_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "test_feature_columns": (
            X_test.columns.tolist() if X_test is not None else None
        ),
        "test_feature_dtypes": (
            X_test.dtypes.astype(str).to_dict()
            if X_test is not None
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "scikit-optimize": _package_version("scikit-optimize"),
            "xgboost": _package_version("xgboost"),
            "cloudpickle": _package_version("cloudpickle"),
        },
        "extra": {} if extra is None else extra,
    }

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            with gzip.GzipFile(
                fileobj=temporary_file,
                mode="wb",
                compresslevel=6,
            ) as compressed_file:
                cloudpickle.dump(
                    checkpoint,
                    compressed_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(path)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    print(f"Checkpoint saved: {path}")
    print(f"Models saved: {list(results)}")
    return path


def load_model_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    """Load a checkpoint created by save_model_checkpoint.

    Only load checkpoint files that you trust.
    """
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with gzip.open(path, "rb") as file:
        checkpoint = cloudpickle.load(file)

    required_keys = {"format_version", "results"}
    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise ValueError(
            f"Invalid checkpoint; missing keys: {sorted(missing_keys)}"
        )

    if checkpoint["format_version"] != 1:
        raise ValueError(
            "Unsupported checkpoint format: "
            f"{checkpoint['format_version']}"
        )

    print(f"Checkpoint loaded: {path}")
    print(f"Models available: {list(checkpoint['results'])}")
    return checkpoint


def evaluate_model_checkpoint(
    checkpoint_or_path: dict[str, Any] | str | Path,
    *,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Load, validate, and evaluate saved searches.

    If X_test and y_test are omitted, the versions stored in the checkpoint
    are used.
    """
    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = load_model_checkpoint(checkpoint_or_path)
    else:
        checkpoint = checkpoint_or_path

    if X_test is None:
        X_test = checkpoint.get("X_test")

    if y_test is None:
        y_test = checkpoint.get("y_test")

    if X_test is None or y_test is None:
        raise ValueError(
            "X_test and y_test must either be stored in the checkpoint "
            "or supplied to evaluate_model_checkpoint()."
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    expected_columns = checkpoint.get("test_feature_columns")
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in X_test.columns]
        unexpected = [c for c in X_test.columns if c not in expected_columns]

        if missing or unexpected:
            raise ValueError(
                "X_test schema differs from the saved schema. "
                f"Missing columns: {missing}; "
                f"unexpected columns: {unexpected}."
            )

        # Reproduce the training/evaluation column order.
        X_test = X_test.loc[:, expected_columns]

    return evaluate_searches(
        checkpoint["results"],
        X_test,
        y_test,
        models=models,
    )


def verify_model_checkpoint(
    original_results: dict[str, ModelSearchResult],
    path: str | Path,
    X_reference: pd.DataFrame,
) -> None:
    """Confirm that reloaded models produce identical probabilities."""
    checkpoint = load_model_checkpoint(path)
    restored_results = checkpoint["results"]

    if set(original_results) != set(restored_results):
        raise AssertionError(
            "Saved model names do not match the original model names."
        )

    for model_name in original_results:
        original_probabilities = (
            original_results[model_name]
            .estimator
            .predict_proba(X_reference)[:, 1]
        )

        restored_probabilities = (
            restored_results[model_name]
            .estimator
            .predict_proba(X_reference)[:, 1]
        )

        np.testing.assert_array_equal(
            original_probabilities,
            restored_probabilities,
            err_msg=f"Predictions changed for {model_name!r}.",
        )

    print("Checkpoint verified: all predictions are identical.")
