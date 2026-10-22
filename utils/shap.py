from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import expit


def check_shap_additivity(
    shap_values: shap.Explanation,
    model: Any,
    X: np.ndarray,
    selected_features: list | None = None,
    verbose: bool = True,
) -> float:
    """Verify the SHAP additivity property."""
    features = X[selected_features] if selected_features is not None else X

    reconstructed_output = (
        np.asarray(shap_values.base_values).reshape(-1)
        + shap_values.values.sum(axis=1)
    )

    raw_model_output = model.decision_function(features)

    maximum_error = np.max(np.abs(reconstructed_output - raw_model_output))

    if verbose:
        status = "PASSED" if maximum_error < 1e-8 else "FAILED"
        print(f"Additivity check {status}: max error = {maximum_error:.3e}")

    return maximum_error


def check_shap_additivity_proba(
    shap_values: shap.Explanation,
    model: Any,
    X: np.ndarray,
    selected_features: list | None = None,
    verbose: bool = True,
) -> float:
    """Verify SHAP additivity in probability space via the sigmoid link."""
    features = X[selected_features] if selected_features is not None else X

    reconstructed_output = (
        np.asarray(shap_values.base_values).reshape(-1)
        + shap_values.values.sum(axis=1)
    )

    reconstructed_proba = expit(reconstructed_output)

    predicted_proba = model.predict_proba(features)[:, 1]

    maximum_error = np.max(np.abs(reconstructed_proba - predicted_proba))

    if verbose:
        status = "PASSED" if maximum_error < 1e-8 else "FAILED"
        print(f"Additivity check (proba) {status}: max error = {maximum_error:.3e}")

    return maximum_error


def plot_shap_bar(
    shap_values: shap.Explanation,
    color: str = "black",
    max_display: int = 20,
) -> None:
    """Plot SHAP bar chart with custom color styling."""
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False,
    )

    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor(color)

    for text in ax.texts:
        text.set_color(color)

    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.show()


def plot_shap_scatter(
    shap_values: shap.Explanation,
    feature: str,
    color: str = "black",
) -> None:
    """Plot SHAP scatter plot for a single feature with custom point color."""
    shap.plots.scatter(
        shap_values[:, feature],
        show=False,
    )

    ax = plt.gca()
    for collection in ax.collections:
        collection.set_color(color)

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()


def plot_shap_scatter_colored_by_feature(
    shap_values: shap.Explanation,
    feature: str,
    color_feature: str,
    cmap: LinearSegmentedColormap | None = None,
) -> None:
    """Plot SHAP scatter plot for a single feature, colored by another feature's SHAP values."""
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["blue", "white", "red"]
        )
    shap.plots.scatter(
        shap_values[:, feature],
        color=shap_values[:, color_feature],
        cmap=cmap,
        show=False,
    )

    plt.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()