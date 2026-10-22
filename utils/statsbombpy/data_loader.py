"""This module contains functions to fetch and process data from the StatsBomb API."""
import logging
import os
import time

import numpy as np
import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)

# KNOWN_FAILED_MATCH_IDS = {3951486, 3951561}


def _get_credentials() -> dict:
    return {
        "user": os.environ["SB_USERNAME"],
        "passwd": os.environ["SB_PASSWORD"],
    }


# --- Competition / season filtering + match fetching ------------------------

def _resolve_competition_mask(
    competitions: pd.DataFrame, competitions_filter: list[int] | list[str] | None
) -> pd.Series:
    """
    Build a boolean mask from a single filter argument that can be either
    competition_ids (all int) or competition_names (all str). Keeping this to
    one parameter avoids the ambiguity/error surface of accepting both at
    once (e.g. contradictory ids/names being passed together).
    """
    if competitions_filter is None:
        return pd.Series(True, index=competitions.index)

    if all(isinstance(c, int) for c in competitions_filter):
        return competitions["competition_id"].isin(competitions_filter)
    if all(isinstance(c, str) for c in competitions_filter):
        return competitions["competition_name"].isin(competitions_filter)

    raise ValueError(
        "competitions_filter must be all int (competition_ids) or all str "
        "(competition_names), not a mix of both."
    )


def _resolve_season_mask(
    competitions: pd.DataFrame, seasons_filter: list[int] | list[str] | None
) -> pd.Series:
    """
    Build a boolean mask from a single filter argument that can be either
    season_ids (all int) or season_names (all str). Mirrors
    _resolve_competition_mask for the same reason: one parameter, no
    ambiguity/error surface from mixing ids and names.
    """
    if seasons_filter is None:
        return pd.Series(True, index=competitions.index)

    if all(isinstance(s, int) for s in seasons_filter):
        return competitions["season_id"].isin(seasons_filter)
    if all(isinstance(s, str) for s in seasons_filter):
        return competitions["season_name"].isin(seasons_filter)

    raise ValueError(
        "seasons_filter must be all int (season_ids) or all str "
        "(season_names), not a mix of both."
    )


def log_available_competitions_seasons(competitions: pd.DataFrame) -> None:
    """Log the distinct competitions and seasons available from the API."""
    available_competitions = (
        competitions[["competition_id", "competition_name"]]
        .drop_duplicates()
        .sort_values("competition_id")
    )
    available_seasons = (
        competitions[["season_id", "season_name"]]
        .drop_duplicates()
        .sort_values("season_id")
    )
    logger.info(
        "Available competitions:\n%s", available_competitions.to_string(index=False)
    )
    logger.info("Available seasons:\n%s", available_seasons.to_string(index=False))


def select_competition_season_rows(
    competitions: pd.DataFrame,
    competitions_filter: list[int] | list[str] | None = None,
    seasons_filter: list[int] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Vectorized filter over the full competitions table.
 
    Building one combined boolean mask (via .isin()) is O(n_competitions)
    regardless of how many seasons/competitions are requested, versus looping
    per season name and re-slicing the DataFrame each time.
 
    competitions_filter=None means "all competitions"; seasons_filter=None
    means "all seasons". Each accepts either ids (all int) or names (all
    str), not a mix of both.
    """
    mask = _resolve_competition_mask(competitions, competitions_filter)
    mask &= _resolve_season_mask(competitions, seasons_filter)

    selected = competitions[mask]

    if seasons_filter is not None:
        season_col = (
            "season_id" if all(isinstance(s, int) for s in seasons_filter) else "season_name"
        )
        found_seasons = set(selected[season_col])
        for season in seasons_filter:
            if season not in found_seasons:
                logger.warning("Season %s not found in filtered competitions.", season)

    return selected


def get_season_matches(
    creds: dict,
    competitions_filter: list[int] | list[str] | None = None,
    seasons_filter: list[int] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch raw match-level data for the requested seasons/competitions.

    seasons=None fetches all available seasons; competitions_filter=None
    fetches all available competitions. Pass competitions_filter as either a
    list of competition_ids or a list of competition_names to disambiguate
    when multiple competitions share a season_name.
    """
    competitions = sb.competitions(creds=creds)
    log_available_competitions_seasons(competitions)

    selected = select_competition_season_rows(competitions, competitions_filter, seasons_filter)

    if selected.empty:
        logger.warning("No competitions matched the given filters.")
        return pd.DataFrame()

    all_matches = []
    # drop_duplicates guards against the same (competition_id, season_id)
    # pair appearing twice in the competitions table.
    for _, row in selected.drop_duplicates(["competition_id", "season_id"]).iterrows():
        competition_id, season_id = row["competition_id"], row["season_id"]
        try:
            matches = sb.matches(
                competition_id=competition_id, season_id=season_id, creds=creds
            )
            logger.info(
                "Competition %s, season %s: %d matches",
                competition_id, row["season_name"], len(matches),
            )
            matches["season_name"] = row["season_name"]
            matches["competition_id"] = competition_id
            all_matches.append(matches)
        except Exception as e:
            logger.error(
                "Failed to fetch matches for competition %s, season %s: %s",
                competition_id, season_id, e,
            )

    if not all_matches:
        return pd.DataFrame()
    return pd.concat(all_matches, ignore_index=True)


# --- Manager nickname enrichment + column selection --------------------------

def make_nickname(name: str) -> str:
    if pd.isna(name):
        return name
    parts = str(name).split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1]}"


def add_manager_nicknames(matches: pd.DataFrame) -> pd.DataFrame:
    """Fill missing home/away manager nicknames by deriving them from full names."""
    matches = matches.copy()

    for side in ("home", "away"):
        nickname_col = f"{side}_manager_nickname"
        name_col = f"{side}_manager_name"
        mask = matches[nickname_col].isna()
        matches.loc[mask, nickname_col] = matches.loc[mask, name_col].apply(make_nickname)

    return matches


def select_match_columns(matches: pd.DataFrame, include_managers: bool) -> pd.DataFrame:
    cols_to_keep = [
        "match_id", "match_date", "season_name", "competition_id", "home_team", "away_team",
    ]

    if include_managers:
        matches = add_manager_nicknames(matches)
        cols_to_keep += ["home_manager_nickname", "away_manager_nickname"]

    return matches[cols_to_keep]


# --- Event fetching -----------------------------------------------------------

def fetch_events_for_match(
    match_id, creds: dict, max_attempts: int = 3, retry_delay_s: float = 2.0
) -> pd.DataFrame | None:
    """Fetch events for a single match, retrying transient failures."""
    for attempt in range(max_attempts):
        try:
            events = sb.events(match_id=match_id, creds=creds, include_360_metrics=True)
            events["match_id"] = match_id
            return events
        except Exception as e:
            logger.warning("Match %s attempt %d failed: %s", match_id, attempt + 1, e)
            time.sleep(retry_delay_s)
    return None


def fetch_all_events(
    match_ids, creds: dict, pause_between_matches_s: float = 0.3
) -> pd.DataFrame: #tuple[pd.DataFrame, list]:
    """Fetch events for every match ID, tracking which ones ultimately fail."""
    all_events = []
    failed_matches = []

    for match_id in match_ids:
        # if match_id in KNOWN_FAILED_MATCH_IDS:
        #     failed_matches.append(match_id)
        #     continue

        events = fetch_events_for_match(match_id, creds)
        if events is not None:
            all_events.append(events)
        else:
            failed_matches.append(match_id)

        time.sleep(pause_between_matches_s)  # ease off the API

    if failed_matches:
        logger.warning("Could not fetch these match IDs after retries: %s", failed_matches)

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return events#, failed_matches


# --- Post-processing -----------------------------------------------------------

def add_opposition_team(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["opposition_team"] = np.where(
        events["team"] == events["home_team"], events["away_team"], events["home_team"]
    )
    return events


# --- Public entry point --------------------------------------------------------

def get_events(
    competitions_filter: list[int] | list[str] | None = None,
    seasons_filter: list[int] | list[str] | None = None,
    include_managers: bool = False,
    # export_data: bool = False,
) -> pd.DataFrame:
    """
    Get events data from StatsBomb API.

    seasons=None fetches all available seasons; competitions_filter=None
    fetches all available competitions. Pass competitions_filter as either a
    list of competition_ids OR a list of competition_names (not both) to
    scope the request, e.g. when multiple competitions share a season_name.
    Available competitions/seasons are logged at INFO level on every call.
    """
    creds = _get_credentials()
    matches = get_season_matches(creds, competitions_filter, seasons_filter)
    matches = select_match_columns(matches, include_managers)
    events = fetch_all_events(matches["match_id"], creds)
    events = events.merge(matches, on="match_id", how="left")
    events = add_opposition_team(events)

    # if export_data:
    #     events.to_csv("data/events.csv", index=False)

    return events
