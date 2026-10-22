import pandas as pd


def goal_rate(df: pd.DataFrame) -> float:
    return (df["shot_outcome"] == "Goal").mean() if len(df) else float("nan")


def get_goal_summary(goals_dict: dict) -> pd.DataFrame:
    summary = pd.DataFrame({
        "label": goals_dict.keys(),
        "n_shots": [len(df) for df in goals_dict.values()],
        "goal_rate": [goal_rate(df) for df in goals_dict.values()],
    }).set_index("label").sort_values(by="n_shots", ascending=False)

    summary["goal_rate"] = summary["goal_rate"].map("{:.2%}".format)

    return summary
