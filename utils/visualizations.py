from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
# from scipy.stats import gaussian_kde


def get_default_palette(n: int) -> list:
    """
    Generate a list of n visually distinct colors,
    scaling the colormap to the category count.
    """
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("nipy_spectral")
        return [cmap(i / n) for i in range(n)]


def set_plot_properties(
    ax: plt.Axes,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    title_fontsize: int | None = None,
    label_fontsize: int | None = None,
    grid: bool = False,
) -> None:
    """Set title, labels, limits, grid, and legend for a matplotlib Axes."""
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=label_fontsize)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if not grid:
        ax.grid(False)


def plot_grouped_boxplot(
    data: pd.DataFrame,
    grouped_variable: str,
    by_variable: str,
    color: str = "cadetblue",
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a boxplot of a numeric variable grouped by a categorical variable."""
    for col in (grouped_variable, by_variable):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    fig, ax = plt.subplots(figsize=figsize)
    data[[grouped_variable, by_variable]].boxplot(
        by=by_variable, color=color, ax=ax
    )

    fig.suptitle("")

    set_plot_properties(
        ax=ax, x_label=grouped_variable, y_label=by_variable
    )

    return fig, ax


def plot_pie_chart(
    data: pd.DataFrame,
    variable: str,
    colors: list | None = None,
    labels: list | None = None,
    autopct: str = "%1.1f%%",
    figsize: tuple[float, float] = (6, 6),
    dropna: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a pie chart of value counts for a categorical variable."""
    counts = data[variable].value_counts(dropna=dropna)

    if colors is None:
        colors = get_default_palette(len(counts))
    elif len(colors) < len(counts):
        raise ValueError(
            f"Need at least {len(counts)} colors, got {len(colors)}"
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        counts, colors=colors, labels=labels, startangle=90,
        autopct=autopct, textprops={"fontsize": 16}
    )
    ax.legend(counts.index, fontsize=16, bbox_to_anchor=(0.7, 0.9))

    return fig, ax


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
    color_map: str | list | Colormap | None = None,
    mask_upper_triangle: bool = False,
    annot_fontsize: int = 12,
    set_title: bool = True,
    figsize: tuple[float, float] = (30, 15)
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a lower-triangle correlation heatmap."""
    corr = data.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=0) if mask_upper_triangle else None

    if color_map is None:
        color_map = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_fontsize},
        vmin=-1, vmax=1,
        cmap=color_map,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    if set_title:
        ax.set_title(f"Correlation Matrix ({method})", fontsize=14, pad=12)

    fig.tight_layout()

    return fig, ax


def plot_confusion_matrix(
    ax: plt.Axes,
    matrix,
    title: str,
    color_map: str = "Oranges",
) -> plt.Axes:
    """Plot a confusion matrix as an annotated heatmap on the given axes."""
    sns.heatmap(matrix, annot=True, fmt="d", cmap=color_map, ax=ax)

    set_plot_properties(
        ax,
        title=f"{title} Confusion Matrix",
        x_label="Predicted Labels",
        y_label="True Labels",
    )

    return ax


def _stacked_bar(
    ax: plt.Axes,
    categories: list[str],
    values_0: pd.Series,
    values_1: pd.Series,
    labels: list[str],
    colors: list[str],
    title: str,
    y_label: str,
    x_label: str,
    rotation: int,
) -> None:
    """
    Draw a two-segment stacked bar chart with
    a legend, title, and axis labels on the given axes.
    """        
    p0 = ax.bar(categories, values_0, 0.55, color=colors[0])
    p1 = ax.bar(categories, values_1, 0.55, bottom=values_0, color=colors[1])
    ax.legend((p1[0], p0[0]), labels[::-1])
    ax.tick_params(axis="x", rotation=rotation)
    set_plot_properties(ax, title=title, x_label=x_label, y_label=y_label)


def plot_categorical_bar_charts(
    df: pd.DataFrame,
    feature: str,
    target: str,
    colors: list[str] = ["lightgray", "orange"],
    figsize: tuple[float, float] = (12, 5),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot side-by-side frequency and proportion stacked bar charts
    of a binary target across a categorical feature.
    """
    cont_tab = pd.crosstab(df[feature], df[target], margins=True)
    if cont_tab.shape[1] - 1 != 2:
        raise ValueError(f"Expected a binary target, got {cont_tab.shape[1] - 1} classes")

    categories = cont_tab.index[:-1].astype(str)  # exclude "All" row and force categorical x-axis
    counts_0 = cont_tab.iloc[:-1, 0]
    counts_1 = cont_tab.iloc[:-1, 1]
    labels = cont_tab.columns[:-1]  # aligned with counts_0 / counts_1, unlike df[target].unique()

    rotation = 0 if df[feature].dtype == "bool" else 90

    fig, (ax_freq, ax_prop) = plt.subplots(1, 2, figsize=figsize)

    _stacked_bar(
        ax_freq, categories, counts_0, counts_1, labels, colors,
        title="Frequency bar chart", x_label=feature, y_label="Frequency", rotation=rotation,
    )

    total_counts = counts_0 + counts_1
    _stacked_bar(
        ax_prop, categories, counts_0 / total_counts, counts_1 / total_counts, labels, colors,
        title="Proportion bar chart", x_label=feature, y_label="Proportion", rotation=rotation,
    )

    fig.tight_layout()

    return fig, (ax_freq, ax_prop)


# def visualize_dimensionality_reduction(
#     transformation: np.ndarray, targets: pd.Series, predictions: pd.Series | None = None
# ) -> None:
#     if predictions is not None:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#     else:
#         fig, ax1 = plt.subplots(figsize=(6, 6))
#         ax2 = None

#     target_colors = np.array(["gray", "orange"])[targets.astype(int)]
#     ax1.scatter(transformation[:, 0], transformation[:, 1], c=target_colors)
    
#     # Create legend
#     handles = [
#         ax1.scatter([], [], c="gray", label="0"),
#         ax1.scatter([], [], c="orange", label="1")
#     ]
#     ax1.legend(handles=handles, title=targets.name)
#     ax1.set_title("True Targets")

#     if ax2 is not None:
#         pred_colors = np.array(["lightgrey", "indianred"])[predictions.astype(int)]
#         ax2.scatter(transformation[:, 0], transformation[:, 1], c=pred_colors)
        
#         handles = [
#             ax2.scatter([], [], c="lightgrey", label="0"),
#             ax2.scatter([], [], c="indianred", label="1")
#         ]
#         ax2.legend(handles=handles, title="Predicted")
#         ax2.set_title("Predictions")

#     plt.show()


def _scatter_with_binary_legend(
    ax: plt.Axes,
    transformation: np.ndarray,
    labels: pd.Series,
    colors: tuple[str, str],
    legend_title: str,
    title: str,
) -> None:
    """Draw a 2D scatter plot colored by a binary label, with a matching legend."""
    point_colors = np.array(colors)[labels.astype(int)]
    ax.scatter(transformation[:, 0], transformation[:, 1], c=point_colors)

    handles = [
        ax.scatter([], [], c=colors[0], label="0"),
        ax.scatter([], [], c=colors[1], label="1"),
    ]
    ax.legend(handles=handles, title=legend_title)
    set_plot_properties(ax, title=title)


def visualize_dimensionality_reduction(
    transformation: np.ndarray,
    targets: pd.Series,
    predictions: pd.Series | None = None,
    target_colors: tuple[str, str] = ("gray", "orange"),
    prediction_colors: tuple[str, str] = ("lightgrey", "indianred"),
    figsize: tuple[float, float] = (6, 6),
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Plot a 2D dimensionality-reduction scatter colored by true targets,
    optionally alongside predictions.
    """
    if transformation.shape[1] < 2:
        raise ValueError("transformation must have at least 2 columns to plot")

    if predictions is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    _scatter_with_binary_legend(
        ax1, transformation, targets, target_colors,
        legend_title=targets.name, title="True Targets"
    )

    if ax2 is not None:
        _scatter_with_binary_legend(
            ax2, transformation, predictions, prediction_colors,
            legend_title="Predicted", title="Predictions"
        )

    fig.tight_layout()

    axes = (ax1, ax2) if ax2 is not None else (ax1,)
    return fig, axes


def _figsize_for_crop(
    length_lim: tuple[float, float],
    width_lim: tuple[float, float],
    vertical_pitch: bool,
    target_size: float = 10,
) -> tuple[float, float]:
    """
    Compute a figsize whose aspect ratio matches the cropped pitch area, avoiding whitespace padding.
    """
    length_range = length_lim[1] - length_lim[0]
    width_range = width_lim[1] - width_lim[0]

    w_dim, h_dim = (width_range, length_range) if vertical_pitch else (length_range, width_range)

    if w_dim >= h_dim:
        width, height = target_size, target_size * h_dim / w_dim
    else:
        height, width = target_size, target_size * w_dim / h_dim

    return (width, height)


def _prepare_event_data(
    df: pd.DataFrame,
    x: str,
    y: str,
    statsbomb_data: bool = True,
) -> pd.DataFrame:
    """Prepare event data by scaling and inverting coordinates to match StatsBomb orientation."""
    df = df.copy()

    if not {x, y}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {x!r} and {y!r}")

    # Scale to StatsBomb pitch (length=120, width=80)
    if not statsbomb_data:
        df[x] = df[x] * 1.2
        df[y] = df[y] * 0.8

    # Invert y (to match mplsoccer convention)
    df[y] = 80 - df[y]

    return df


def _set_dimensions_lim(
    ax: plt.Axes,
    vertical_pitch: bool,
    length_lim: tuple[float, float],
    width_lim: tuple[float, float],
) -> None:
    """Apply the pitch-length axis limit to the correct axis depending on pitch orientation."""
    if vertical_pitch:
        ax.set_ylim(length_lim[0], length_lim[1])
        ax.set_xlim(width_lim[0], width_lim[1])
    else:
        ax.set_xlim(length_lim[0], length_lim[1])
        ax.set_ylim(width_lim[0], width_lim[1])


def _bottom_left_text_pos(
    vertical_pitch: bool,
    length_lim: tuple[float, float],
    width_lim: tuple[float, float],
    line_index: int = 0,
    line_gap: float = 3,
) -> tuple[float, float]:
    """
    Compute a data-coordinate anchor near the pitch's bottom-left corner,
    stacking lines upward as line_index increases.
    """
    if vertical_pitch:
        # x = width axis (screen-horizontal), y = length axis (screen-vertical)
        x = width_lim[0] + 3
        y = length_lim[0] + 1 + line_index * line_gap
    else:
        # x = length axis (screen-horizontal), y = width axis (screen-vertical)
        x = length_lim[0] + 3
        y = width_lim[0] + 1 + line_index * line_gap
    return x, y


def _plot_standard(
    df: pd.DataFrame,
    x: str,
    y: str,
    target: str | None,
    pitch: Pitch | VerticalPitch,
    vertical_pitch: bool = False,
    length_lim: tuple[float, float] = (0, 120),
    width_lim: tuple[float, float] = (0, 80),
    colors: tuple[str, str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a standard scatter pitch map, optionally split and colored by a binary target."""
    figsize = _figsize_for_crop(length_lim, width_lim, vertical_pitch)
    fig, ax = pitch.draw(figsize=figsize)

    _set_dimensions_lim(ax, vertical_pitch, length_lim, width_lim)

    text_bbox = {"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3}

    title = "Pitch Map"
    total = len(df)

    if target:
        target_bool = df[target].astype(bool)

        title += f" - {target}".upper()
        count = target_bool.sum()
        rate = 100 * count / total

        colors = colors or ("grey", "orange")

        for cat, color in zip([False, True], colors):
            subset = df[target_bool == cat]
            pitch.scatter(
                subset[x], subset[y], ax=ax,
                color=color, s=50, alpha=0.5, zorder=2, label=str(cat),
                edgecolors="black" if cat else None,
            )

        ax.legend(loc="upper right")

        rate_x, rate_y = _bottom_left_text_pos(vertical_pitch, length_lim, width_lim, line_index=1)
        ax.text(
            rate_x, rate_y, f"{target.capitalize()} Rate: {rate:.2f}% ({count})",
            color="black", fontsize=12, weight="bold", bbox=text_bbox,
        )
    else:
        pitch.scatter(
            df[x], df[y], ax=ax, c=colors[0], s=50, alpha=0.5, zorder=2, edgecolors="black"
        )

    ax.set_title(title, fontsize=14, weight="bold")

    total_x, total_y = _bottom_left_text_pos(vertical_pitch, length_lim, width_lim, line_index=0)
    ax.text(
        total_x, total_y, f"Total events: {total}", 
        color="black", fontsize=12, weight="bold", bbox=text_bbox
    )

    return fig, ax


# def _get_max_kde_density(df: pd.DataFrame, x: str, y: str, target: str) -> float:
#     """Compute the global maximum density across all target subsets."""
#     max_density = 0.0
#     for cat in df[target].dropna().unique():
#         sub = df[df[target] == cat]
#         if len(sub) > 1:
#             kde = gaussian_kde(np.vstack([sub[x], sub[y]]))
#             # Evaluate over the data range
#             eval_x = np.linspace(sub[x].min(), sub[x].max(), 50)
#             eval_y = np.linspace(sub[y].min(), sub[y].max(), 50)
#             xx, yy = np.meshgrid(eval_x, eval_y)
#             densities = kde(np.vstack([xx.ravel(), yy.ravel()]))
#             max_density = max(max_density, densities.max())
#     return max_density


def _plot_heatmap(  # TO BE IMPROVED (MAYBE USE BINS INSTEAD OF KDE)
    df: pd.DataFrame,
    x: str,
    y: str,
    target: str | None,
    pitch: Pitch | VerticalPitch,
    vertical_pitch: bool = False,
    length_lim: tuple[float, float] = (0, 120),
    width_lim: tuple[float, float] = (0, 80),
    colors: tuple[str, str] | None = None,
    # bins: tuple[int, int] = (25, 25),
) -> tuple[plt.Figure, object]:
    """Plot heatmap(s) of events, split by a binary target if one is given."""
    if target:
        categories = df[target].dropna().unique()
        print(categories)
        n = len(categories)

        # subsets = [df[df[target] == cat] for cat in categories]
        # bin_stats = [
        #     pitch.bin_statistic(
        #         sub[x], sub[y], statistic="count", bins=bins
        #     )
        #     for sub in subsets
        # ]
        # global_vmax = max(
        #     (bs["statistic"].max() for bs in bin_stats if bs["statistic"].size > 0),
        #     default=1,
        # )

        panel_w, panel_h = _figsize_for_crop(length_lim, width_lim, vertical_pitch)
        fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h))
        # fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h), squeeze=False)
        # axes_flat = axes.flatten()

        cmaps = colors or ("grey", "orange")

        for i, (cat, cmap) in enumerate(zip(categories, cmaps)):
        # for i, (cat, subset, bs) in enumerate(zip(categories, subsets, bin_stats)):
            subset = df[df[target] == cat]
            # ax = axes_flat[i]

            pitch.draw(ax=axes[i])
            # pitch.draw(ax=ax)

            pitch.kdeplot(
                subset[x], subset[y], ax=axes[i], fill=True,
                thresh=0.01, levels=100, cmap=cmap,
            )

            # if not subset.empty:
            #     pitch.heatmap(
            #         bs,
            #         ax=ax,
            #         cmap=cmap,
            #         edgecolors="none",
            #         vmin=0,
            #         vmax=global_vmax,
            #     )

            _set_dimensions_lim(axes[i], vertical_pitch, length_lim, width_lim)

            axes[i].text(
                40, 125, f"{target.title()} = {cat}",
                fontsize=14, color="black", ha="center", weight="bold",
            )
            axes[i].text(
                40, 123, f"Events: {len(subset)}",
                fontsize=12, color="darkgrey", ha="center", weight="bold",
            )

        fig.suptitle(f"{target.title()} Heatmaps", fontsize=24, color="#000009")
        fig.tight_layout()

        return fig, axes

    fig, ax = pitch.draw(figsize=_figsize_for_crop(length_lim, width_lim, vertical_pitch))
    pitch.kdeplot(df[x], df[y], ax=ax, fill=True, thresh=0.01, levels=100, cmap=cmaps[0])
    _set_dimensions_lim(ax, vertical_pitch, length_lim, width_lim)
    ax.set_title("Pitch Heatmap", fontsize=14, weight="bold")

    return fig, ax


def plot_pitch_map(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    target: str | None = None,
    map_type: Literal["standard", "heat"] = "standard",
    vertical_pitch: bool = False,
    invert_pitch_colors: bool = False,
    colors: tuple[str, str] | None = None,
    statsbomb_data: bool = True,
    length_lim: tuple = (0, 120),
    width_lim: tuple = (0, 80),
) -> tuple[plt.Figure, object]:
    """Plot an event map (scatter or heatmap) on a football pitch, vertical or horizontal."""
    if target is not None and target not in df.columns:
        raise ValueError(f"Column '{target}' not found in data")

    df = _prepare_event_data(df, x, y, statsbomb_data)

    pitch_color, line_color = ("black", "white") if invert_pitch_colors else ("white", "black")
    pitch = VerticalPitch if vertical_pitch else Pitch
    pitch = pitch(
        pitch_type="statsbomb", pitch_color=pitch_color, line_color=line_color, line_zorder=3
    )

    if map_type == "standard":
        return _plot_standard(
            df, x, y, target, pitch, vertical_pitch, length_lim, width_lim, colors
        )
    elif map_type == "heat":
        return _plot_heatmap(
            df, x, y, target, pitch, vertical_pitch, length_lim, width_lim, colors
        )
    else:
        raise ValueError("map_type must be either 'standard' or 'heat'")


def plot_calibration_and_distribution(
    y_test: pd.Series,
    test_predictions: dict[str, np.ndarray],
    colors: dict[str, str] | None = None,
    n_bins: int = 8,
    strategy: str = "quantile",
    hist_bins: np.ndarray | None = None,
    figsize: tuple[float, float] = (13, 5),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot calibration curves and predicted-probability distributions for one or more models."""
    if hist_bins is None:
        hist_bins = np.linspace(0, 1, 31)

    if colors is None:
        palette = get_default_palette(len(test_predictions))
        colors = dict(zip(test_predictions.keys(), palette))
    elif set(colors) != set(test_predictions):
        raise ValueError(
            f"colors keys {set(colors)} must match test_predictions keys {set(test_predictions)}"
        )

    fig, (ax_calib, ax_dist) = plt.subplots(1, 2, figsize=figsize)

    for model_name, probability in test_predictions.items():
        color = colors[model_name]
        CalibrationDisplay.from_predictions(
            y_test, probability, n_bins=n_bins, strategy=strategy,
            name=model_name, ax=ax_calib, color=color,
        )
        ax_dist.hist(
            probability, bins=hist_bins, histtype="step",
            linewidth=1.5, label=model_name, color=color,
        )

    set_plot_properties(ax_calib, title="Test-set calibration")
    set_plot_properties(
        ax_dist, title="Distribution of predicted xG",
        x_label="Predicted xG", y_label="Number of shots",
        x_lim=(0, 1),
    )
    ax_dist.legend()

    fig.tight_layout()

    return fig, (ax_calib, ax_dist)
