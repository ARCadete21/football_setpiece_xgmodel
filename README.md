# Set-Piece xG Model

An expected-goals (xG) model for shots following set-piece routines, built with StatsBomb event and shot freeze-frame data.

Version 1 covers **first-phase shots following corners and indirect free kicks**. Penalties, direct free kicks, throw-ins, and second-phase situations are outside its scope.

## Project structure

| Path | Purpose |
| --- | --- |
| `notebook.ipynb` | Data exploration, feature engineering, model selection, evaluation, calibration assessment, and interpretation. |
| `script.py` | Python API and command-line interface for scoring new event datasets. |
| `utils/` | Feature extraction, data loading, visualization, modeling, evaluation, and checkpoint utilities. |
| `models/pipeline.joblib` | Exported fitted pipeline used for inference. |
| `requirements.txt` | Pinned package versions from the development environment, including indirect dependencies. |

## Installation

The project was developed with **Python 3.12.6**. Use this version and the supplied package versions to reproduce the development environment.

Run the following commands from the project root.

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check the interpreter version before installing dependencies:

```bash
python --version
```

It should report `Python 3.12.6`; the commands above select an installed interpreter and do not install that Python version.

## Input data

The inference module accepts a pandas DataFrame or CSV containing **flattened StatsBomb events**, using the same schema and coordinate system as the training notebook. It does not convert raw nested event JSON or calculate `set_piece_phase`.

### Required columns

| Column | Expected content |
| --- | --- |
| `type` | Event type; eligible events must be `Shot`. |
| `play_pattern` | Eligible values: `From Corner` or `From Free Kick`. |
| `shot_type` | Eligible shots must have the value `Open Play`. |
| `set_piece_phase` | Eligible shots must have phase `1`. |
| `location` | Shot coordinates in the StatsBomb coordinate system. |
| `under_pressure` | Whether the shooter was under pressure. |
| `shot_first_time` | Whether the shot was taken first time. |
| `shot_one_on_one` | Whether the shot was a one-on-one opportunity. |
| `shot_open_goal` | Whether the shot was taken at an open goal. |
| `shot_follows_dribble` | Whether the shot followed a dribble. |
| `shot_aerial_won` | Whether the shot followed an aerial duel won by the shooter. |
| `shot_freeze_frame` | Shot freeze frame containing player locations, teammate flags, and position metadata. |
| `shot_body_part` | Body part used for the shot. |
| `shot_technique` | Shot technique. |
| `shot_key_pass_id` | Key-pass identifier, or a missing value when no key pass is recorded. |

All 15 columns are required, including the six boolean context columns. A missing column raises a `ValueError`, even when the dataset contains no eligible shots. The DataFrame index must be unique. The table follows the declaration order shown in `REQUIRED_COLUMNS`; the input DataFrame itself does not need to use that column order.

`shot_type="Open Play"` describes the finish within the set-piece possession; `play_pattern` identifies the origin of that possession.

### Missing columns and missing values

A required column must exist in the input DataFrame. This does not mean that every value in that column must be populated.

If any required column is absent, whether boolean or non-boolean, `add_set_piece_xg` raises a `ValueError` listing the missing columns before loading the model or making predictions. This check also applies to datasets with no eligible shots.

Missing values within existing columns are handled differently:

| Column(s) | Treatment of missing values |
| --- | --- |
| `under_pressure`, `shot_first_time`, `shot_one_on_one`, `shot_open_goal`, `shot_follows_dribble`, `shot_aerial_won` | For eligible shots, converted to `False` by `_as_bool`. Columns are not automatically created when absent. |
| `shot_body_part` | For eligible shots, replaced with `Other`. |
| `shot_technique` | For eligible shots, grouped into `Other`. |
| `shot_key_pass_id` | For eligible shots, produces `shot_assisted=False`. |
| `location` | For eligible shots, a missing location or one that is not a list with at least two elements raises an error. Coordinate values must also be usable in the geometry calculations. |
| `shot_freeze_frame` | For eligible shots, a missing frame produces no geometric features for that row. Mixed batches may still be scored with missing feature values; if every eligible shot lacks a frame, feature alignment raises an error. See the freeze-frame limitations below. |
| `type`, `play_pattern`, `shot_type`, `set_piece_phase` | Missing values do not satisfy the eligibility filter, so the event receives `NaN` in `shot_set_piece_xg`. Non-numeric phase values are also treated as ineligible. |

Feature preparation is applied only to eligible shots. The boolean conversions and categorical replacements affect the model-input features; the returned event DataFrame preserves the original input values in those columns.

### Coordinates and nested fields

`location` and `shot_freeze_frame` may contain Python objects, JSON strings, or Python-literal strings loaded from CSV.

Use StatsBomb's 120 × 80 coordinate system and its attacking orientation towards `x=120`. Geometry features are calculated in metres using an assumed 105 × 68 m pitch. Do not pass coordinates already converted to metres.

The notebook uses three-component shot locations `[x, y, z]`. The inference script also accepts `[x, y]`, assigning `z=0`.

Use valid, populated shot freeze frames for reliable application within the training conditions. Missing freeze-frame handling is currently limited: a batch in which every eligible shot lacks a freeze frame raises a feature-schema error; mixed batches may still produce predictions for shots with missing geometric information. An empty list is treated as a frame with no observed players, rather than missing information.

Neither `shot_outcome` nor `shot_statsbomb_xg` is required for inference.

## Export the fitted model

The supplied model is located at `models/pipeline.joblib`. After retraining and fitting `final_pipeline` in the notebook, export the updated pipeline:

```python
from pathlib import Path

import joblib

model_path = Path("models/pipeline.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_pipeline, model_path)
```

The exported pipeline contains fitted preprocessing, the fixed feature selector, and the classifier. It expects engineered features; `script.py` constructs those features from the input events before calling the pipeline.

Keep the model and feature-extraction code together. Saving only the classifier is insufficient.

## Apply the model in Python

Run from the project root:

```python
import pandas as pd

from script import add_set_piece_xg

events = pd.read_csv("data/new_events.csv")

scored_events = add_set_piece_xg(
    events,
    model="models/pipeline.joblib",
)

scored_events.to_csv(
    "data/new_events_with_set_piece_xg.csv",
    index=False,
)
```

By default, the function returns a copy of the complete event dataset with a `shot_set_piece_xg` column:

- Eligible shots receive a scoring probability between 0 and 1.
- Events outside the model's scope receive `NaN`.
- Original rows and the DataFrame index are preserved in memory. The CSV example omits the index when saving.

An xG value of `0.10` represents an estimated 10% probability of scoring. It is a probability estimate, rather than a goal/non-goal classification.

The function resets `shot_set_piece_xg` if that column already exists. It also accepts a loaded fitted pipeline in the `model` argument, allowing repeated calls without loading the file each time.

## Command-line usage

```bash
python script.py data/new_events.csv models/pipeline.joblib data/new_events_with_set_piece_xg.csv
```

The three positional arguments are the input CSV, fitted-pipeline file, and output CSV, respectively. The command creates missing parent directories for the output file.

## Model and evaluation

The selected classifier is **HistGradientBoostingClassifier**. The final pipeline receives 46 engineered features and retains 15 predictors describing shot context, geometry, nearby defenders, and goalkeeper positioning.

The development dataset contains **3,685 eligible shots** from the **2023/2024, 2024/2025, and 2025/2026** seasons of the Portuguese First Division. Evaluation used a stratified random train–test split, with 737 shots and 64 goals in the test set.

The model achieved performance comparable to the StatsBomb xG benchmark. Differences across the reported metrics were small, and paired bootstrap confidence intervals included zero. These results support the feasibility of a specialized set-piece model but do not establish consistent superiority over StatsBomb.

The final deployment pipeline is fitted on the full dataset. The `shot_set_piece_xg` column exported by the notebook's cross-validation step contains out-of-fold predictions. Applying the deployment pipeline to the same training events will therefore not reproduce that column exactly.

## Scope and limitations

The model has not been externally validated across other competitions or football environments. Its small dataset limits the precision of performance and calibration estimates, particularly for rare situations.

Random splitting does not account for dependence between shots from the same match, team, or routine. Future evaluation should include grouped and temporal validation, external competitions, and match-level uncertainty estimates.

The eligibility filter uses the supplied event labels. It does not reconstruct the restart sequence or independently verify that a possession represents an indirect routine. The notebook identifies rebounds following failed direct free kicks as a case requiring further review.

Freeze-frame features represent the moment of the shot and do not capture the full development of the routine. Missing or incomplete observations can affect predictions.

## Reproducibility

Use Python 3.12.6 and install the versions in `requirements.txt`. The file records the supplied development-environment package list; it is not a minimal list of project dependencies or a cross-platform lockfile.

Preserve the same feature-extraction code, coordinate conventions, category definitions, and fitted pipeline for training and inference. Changes to these components require compatibility checks and may require retraining and exporting a new model.

The inference script checks required columns, index uniqueness, basic shot-location structure, and the pipeline's expected feature names and order. These checks do not provide complete validation of event semantics, coordinate ranges, or freeze-frame quality.

Keep model versions and dependency versions alongside generated results so predictions can be traced to the environment and model that produced them.
