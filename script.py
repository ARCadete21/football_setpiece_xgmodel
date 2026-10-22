"""Fast inference for the first-version StatsBomb set-piece xG model.

The fitted ``final_pipeline`` from the training notebook must be exported once::

    import joblib
    joblib.dump(final_pipeline, "models/pipeline.joblib")

This module preserves every input event and assigns a probability only to the
shots covered by version 1 of the model: first-phase corners and indirect free
kicks. All other rows receive ``NaN`` in ``shot_set_piece_xg``.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.statsbombpy.data_transformer import extract_freeze_frame_features


OUTPUT_COLUMN = "shot_set_piece_xg"

BOOLEAN_COLUMNS = (
    "under_pressure",
    "shot_first_time",
    "shot_one_on_one",
    "shot_open_goal",
    "shot_follows_dribble",
    "shot_aerial_won",
)

REQUIRED_COLUMNS = {
    *BOOLEAN_COLUMNS,
    "type",
    "play_pattern",
    "shot_type",
    "set_piece_phase",
    "location",
    "shot_freeze_frame",
    "shot_body_part",
    "shot_technique",
    "shot_key_pass_id",
}

BODY_PART_LEVELS = ("Head", "Left Foot", "Other", "Right Foot")
TECHNIQUE_LEVELS = ("Half Volley", "Normal", "Other", "Volley")


def _parse_nested(value: Any, *, column: str) -> Any:
    """Parse list/dict values stored as JSON or Python-literal strings."""
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{column} contains an unsupported value: {value!r}")

    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Could not parse {column} value: {value[:100]!r}") from exc


def _as_bool(series: pd.Series) -> pd.Series:
    """Match the notebook convention: missing boolean flags mean False."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    true_values = {True, 1, "1", "true", "True", "TRUE"}
    return series.map(lambda value: value in true_values if pd.notna(value) else False).astype(bool)


def _bin_shot_technique(value: Any) -> str:
    if value in {"Normal", "Volley", "Half Volley"}:
        return str(value)
    return "Other"


def _eligible_shots(events: pd.DataFrame) -> pd.Series:
    """Return the exact population used to train model version 1."""
    is_shot = events["type"].eq("Shot")
    is_open_play_shot = events["shot_type"].eq("Open Play")
    is_supported_restart = events["play_pattern"].isin(
        ["From Corner", "From Free Kick"]
    )
    is_phase_one = pd.to_numeric(
        events["set_piece_phase"], errors="coerce"
    ).eq(1)
    return is_shot & is_open_play_shot & is_supported_restart & is_phase_one


def _one_hot_fixed_levels(
    frame: pd.DataFrame,
    source_column: str,
    levels: tuple[str, ...],
) -> None:
    """Create the fixed dummy columns used by the training notebook."""
    for level in levels:
        frame[f"{source_column}_{level}"] = frame[source_column].eq(level)
    frame.drop(columns=source_column, inplace=True)


def prepare_model_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Convert eligible flattened StatsBomb shot events into model features."""
    data = shots.copy()

    for column in BOOLEAN_COLUMNS:
        data[column] = _as_bool(data[column])

    data["location"] = data["location"].map(
        lambda value: _parse_nested(value, column="location")
    )
    data["shot_freeze_frame"] = data["shot_freeze_frame"].map(
        lambda value: _parse_nested(value, column="shot_freeze_frame")
    )

    invalid_location = data["location"].map(
        lambda value: not isinstance(value, list) or len(value) < 2
    )
    if invalid_location.any():
        bad = data.index[invalid_location].tolist()[:5]
        raise ValueError(f"Eligible shots have invalid locations at indices {bad}.")

    data["shot_from_corner"] = data["play_pattern"].eq("From Corner")
    data["shot_assisted"] = data["shot_key_pass_id"].notna()

    freeze_features = data.apply(
        lambda row: extract_freeze_frame_features(
            shot_location=row["location"],
            freeze_frame=row["shot_freeze_frame"],
        ),
        axis=1,
        result_type="expand",
    )
    data = pd.concat([data, freeze_features], axis=1)

    data["x"] = data["location"].map(lambda value: float(value[0]))
    data["y"] = data["location"].map(lambda value: float(value[1]))
    data["z"] = data["location"].map(
        lambda value: float(value[2]) if len(value) > 2 else 0.0
    )

    data["shot_technique"] = data["shot_technique"].map(_bin_shot_technique)
    data["shot_body_part"] = data["shot_body_part"].fillna("Other")
    _one_hot_fixed_levels(data, "shot_body_part", BODY_PART_LEVELS)
    _one_hot_fixed_levels(data, "shot_technique", TECHNIQUE_LEVELS)

    keep = {
        *BOOLEAN_COLUMNS,
        "shot_from_corner",
        "shot_assisted",
        "x",
        "y",
        "z",
        *(f"shot_body_part_{level}" for level in BODY_PART_LEVELS),
        *(f"shot_technique_{level}" for level in TECHNIQUE_LEVELS),
        *freeze_features.columns,
    }
    return data.loc[:, [column for column in data.columns if column in keep]]


def _align_to_pipeline(features: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """Enforce the feature names and order recorded by scikit-learn."""
    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is None:
        raise TypeError(
            "The model artifact has no feature_names_in_. Export the fitted "
            "final_pipeline from the notebook, not only the classifier."
        )

    expected = list(expected)
    missing = [column for column in expected if column not in features]
    if missing:
        raise ValueError(
            "Could not construct all features expected by the fitted pipeline: "
            + ", ".join(missing)
        )
    return features.loc[:, expected]


def add_set_piece_xg(
    events: pd.DataFrame,
    model: str | Path | Any,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Return StatsBomb events with an out-of-scope-aware set-piece xG column.

    Parameters
    ----------
    events:
        Flattened StatsBomb events in the same schema used by the notebook.
    model:
        A fitted ``final_pipeline`` or the path to its joblib artifact.
    copy:
        If True (default), do not mutate the caller's DataFrame.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(events.columns))
    if missing:
        raise ValueError("Missing required StatsBomb columns: " + ", ".join(missing))
    if not events.index.is_unique:
        raise ValueError("The events DataFrame index must be unique.")

    result = events.copy() if copy else events
    result[OUTPUT_COLUMN] = np.nan
    eligible = _eligible_shots(result)
    if not eligible.any():
        return result

    pipeline = joblib.load(model) if isinstance(model, (str, Path)) else model
    features = prepare_model_features(result.loc[eligible])
    features = _align_to_pipeline(features, pipeline)
    probabilities = pipeline.predict_proba(features)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("The fitted pipeline returned non-finite probabilities.")
    result.loc[eligible, OUTPUT_COLUMN] = probabilities
    return result


def score_csv(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load, score, and save a flattened StatsBomb events CSV."""
    events = pd.read_csv(input_path)
    scored = add_set_piece_xg(events, model_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add first-phase corner/indirect-free-kick xG to StatsBomb events."
    )
    parser.add_argument("input", help="Input flattened StatsBomb events CSV")
    parser.add_argument("model", help="Fitted final_pipeline joblib file")
    parser.add_argument("output", help="Output CSV")
    args = parser.parse_args()
    output = score_csv(args.input, args.model, args.output)
    print(output)


if __name__ == "__main__":
    main()
