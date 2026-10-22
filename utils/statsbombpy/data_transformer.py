from __future__ import annotations

from dataclasses import dataclass
from logging import config
from typing import Any, Sequence

import numpy as np




# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FreezeFrameConfig:
    """Geometry/configuration used for StatsBomb freeze-frame features."""

    sb_pitch_length: float = 120.0
    sb_pitch_width: float = 80.0

    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    goal_x: float = 120.0
    goal_y_min: float = 36.0
    goal_y_max: float = 44.0

    defender_radius_m: float = 0.35
    goalkeeper_radius_m: float = 0.50


CONFIG = FreezeFrameConfig()


# ---------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------

def statsbomb_to_metres(
    location: Sequence[float],
    config: FreezeFrameConfig = CONFIG,
) -> np.ndarray:
    """
    Convert the x/y components of StatsBomb coordinates to metres.

    Any additional dimensions (e.g. z for shot locations) are ignored.
    """
    if location is None or len(location) < 2:
        raise ValueError("Location must contain at least x and y coordinates.")

    x, y = location[:2]

    return np.array(
        [
            x * config.pitch_length_m / config.sb_pitch_length,
            y * config.pitch_width_m / config.sb_pitch_width,
        ],
        dtype=float,
    )


def euclidean_distance(
    p1: np.ndarray,
    p2: np.ndarray,
) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(p1 - p2))


def point_in_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return True when `point` lies inside or on the boundary of
    triangle (a, b, c).

    Uses barycentric coordinates.
    """
    v0 = c - a
    v1 = b - a
    v2 = point - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denominator = dot00 * dot11 - dot01 * dot01

    if abs(denominator) < tolerance:
        return False

    inv_denom = 1.0 / denominator

    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (
        u >= -tolerance
        and v >= -tolerance
        and (u + v) <= 1.0 + tolerance
    )


def distance_point_to_line(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    """
    Perpendicular distance from a point to the infinite line
    passing through line_start and line_end.
    """
    line_vector = line_end - line_start
    line_length = np.linalg.norm(line_vector)

    if line_length == 0:
        return np.nan

    point_vector = point - line_start

    cross = abs(
        line_vector[0] * point_vector[1]
        - line_vector[1] * point_vector[0]
    )

    return float(cross / line_length)


# ---------------------------------------------------------------------
# Goal geometry
# ---------------------------------------------------------------------

def get_goal_points(
    config: FreezeFrameConfig = CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return metric coordinates for:
        lower post, upper post, goal centre
    """
    post_1 = statsbomb_to_metres(
        [config.goal_x, config.goal_y_min],
        config,
    )

    post_2 = statsbomb_to_metres(
        [config.goal_x, config.goal_y_max],
        config,
    )

    goal_center = statsbomb_to_metres(
        [
            config.goal_x,
            (config.goal_y_min + config.goal_y_max) / 2,
        ],
        config,
    )

    return post_1, post_2, goal_center


def signed_angle_from_centerline(
    ball: np.ndarray,
    target: np.ndarray,
    goal_center: np.ndarray,
) -> float:
    """
    Signed angle between:

        ball -> goal centre
        ball -> target

    Returns radians in [-pi, pi].

    Using goal-centred relative angles avoids problems associated
    with absolute pitch orientation.
    """
    center_vector = goal_center - ball
    target_vector = target - ball

    cross = (
        center_vector[0] * target_vector[1]
        - center_vector[1] * target_vector[0]
    )

    dot = np.dot(center_vector, target_vector)

    return float(np.arctan2(cross, dot))


def goal_angle_interval(
    ball: np.ndarray,
    post_1: np.ndarray,
    post_2: np.ndarray,
    goal_center: np.ndarray,
) -> tuple[float, float]:
    """Angular interval occupied by the goal from the ball's perspective."""
    angle_1 = signed_angle_from_centerline(ball, post_1, goal_center)
    angle_2 = signed_angle_from_centerline(ball, post_2, goal_center)
    return min(angle_1, angle_2), max(angle_1, angle_2)


# ---------------------------------------------------------------------
# Angular obstruction
# ---------------------------------------------------------------------

def angular_interval_for_player(
    ball: np.ndarray,
    player: np.ndarray,
    goal_center: np.ndarray,
    radius_m: float,
) -> tuple[float, float] | None:
    """
    Approximate a player as a circle.

    Returns the angular interval subtended by that circle from
    the ball's perspective.
    """
    distance = euclidean_distance(ball, player)

    if distance <= 0:
        return None

    player_angle = signed_angle_from_centerline(ball, player, goal_center)

    ratio = min(radius_m / distance, 1.0)
    half_angle = np.arcsin(ratio)

    return (player_angle - half_angle, player_angle + half_angle)


def clip_interval(
    interval: tuple[float, float],
    limits: tuple[float, float],
) -> tuple[float, float] | None:
    """Clip an angular interval to another angular interval."""
    start = max(interval[0], limits[0])
    end = min(interval[1], limits[1])

    if end <= start:
        return None

    return start, end


def union_interval_length(
    intervals: list[tuple[float, float]],
) -> float:
    """
    Calculate total length of the union of 1-D intervals.

    This prevents overlapping defenders from being double-counted.
    """
    if not intervals:
        return 0.0

    intervals = sorted(intervals, key=lambda x: x[0])

    total = 0.0
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end

    total += current_end - current_start

    return float(total)


def calculate_angle_coverage(
    ball: np.ndarray,
    players: list[np.ndarray],
    goal_interval: tuple[float, float],
    goal_center: np.ndarray,
    radius_m: float,
) -> float:
    """
    Calculate how many radians of the visible goal are covered
    by one or more players.
    """
    clipped_intervals = []

    for player in players:
        interval = angular_interval_for_player(
            ball=ball,
            player=player,
            goal_center=goal_center,
            radius_m=radius_m,
        )

        if interval is None:
            continue

        clipped = clip_interval(interval, goal_interval)

        if clipped is not None:
            clipped_intervals.append(clipped)

    return union_interval_length(clipped_intervals)


# ---------------------------------------------------------------------
# Freeze-frame parsing
# ---------------------------------------------------------------------

def is_goalkeeper(player: dict[str, Any]) -> bool:
    """Identify goalkeeper using StatsBomb position id."""
    position = player.get("position") or {}
    return position.get("id") == 1


def parse_freeze_frame(
    freeze_frame: list[dict[str, Any]],
    config: FreezeFrameConfig = CONFIG,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray | None,
]:
    """
    Extract:
        - opposition outfield defenders
        - opposition goalkeeper

    All returned coordinates are in metres.
    """
    defenders = []
    teammates = []
    goalkeeper = None

    for player in freeze_frame:
        location = player.get("location")

        if location is None or len(location) < 2:
            continue

        location_m = statsbomb_to_metres(
            location,
            config,
        )

        if player.get("teammate", False):
            teammates.append(location_m)
            continue

        if is_goalkeeper(player):
            goalkeeper = location_m
        else:
            defenders.append(location_m)

    return defenders, teammates, goalkeeper


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def nearest_distance_features(
    ball: np.ndarray,
    players: list[np.ndarray],
    n_nearest: int,
    prefix: str,
) -> dict[str, float]:
    """Generate distance features for the N players nearest to the ball."""
    if n_nearest <= 0:
        return {}

    distances = sorted(
        euclidean_distance(ball, player)
        for player in players
    )

    return {
        f"{prefix}_{rank}_distance": (
            distances[rank - 1]
            if len(distances) >= rank
            else np.nan
        )
        for rank in range(1, n_nearest + 1)
    }


def within_distance_features(
    ball: np.ndarray,
    players: list[np.ndarray],
    thresholds_m: Sequence[float] | None,
    prefix: str,
) -> dict[str, int]:
    """Count players within each specified distance of the ball."""
    if thresholds_m is None:
        return {}

    distances = [euclidean_distance(ball, player) for player in players]

    features = {}

    for threshold in thresholds_m:
        if threshold < 0:
            raise ValueError(
                "Distance thresholds must be >= 0."
            )

        threshold_label = (
            str(int(threshold))
            if float(threshold).is_integer()
            else str(threshold).replace(".", "_")
        )

        features[f"{prefix}_{threshold_label}m"] = sum(
            distance <= threshold
            for distance in distances
        )

    return features


def players_in_triangle(
    players: list[np.ndarray],
    ball: np.ndarray,
    post_1: np.ndarray,
    post_2: np.ndarray,
) -> list[np.ndarray]:
    """Return players located inside or on the shot triangle."""
    return [
        player for player in players
        if point_in_triangle(player, ball, post_1, post_2)
    ]


# ---------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------

def extract_freeze_frame_features(
    shot_location: Sequence[float],
    freeze_frame: list[dict[str, Any]] | None,
    *,
    n_nearest_players: int = 3,
    distance_thresholds_m: Sequence[float] | None = (1, 3, 5),
    include_triangle_vars: bool = True,
    include_goal_angle_vars: bool = True,
    include_goalkeeper_vars: bool = True,
    include_teammates_vars: bool = True,
    config: FreezeFrameConfig = CONFIG,
) -> dict[str, float]:
    """
    Extract configurable geometry-based features from a StatsBomb
    shot freeze frame.

    Parameters
    ----------
    shot_location:
        Shot [x, y] in StatsBomb coordinates.

    freeze_frame:
        StatsBomb shot_freeze_frame object.

    nearest_players:
        Number of nearest-player distance features to generate.

        This applies to defenders and, when include_teammates=True,
        to teammates too.

        If 0, nearest-distance features are omitted.

    distance_thresholds_m:
        Distance thresholds used for "within Xm" features.

        This applies to defenders and, when include_teammates=True,
        to teammates too.

        If None, these features are omitted.

    include_triangle_vars:
        Whether to calculate shot-triangle features.

    include_goal_angle_vars:
        Whether to calculate defensive goal-angle obstruction features.

    include_goalkeeper_vars:
        Whether to calculate goalkeeper features.

    include_teammates_vars:
        Whether to calculate teammate-related features.

    config:
        FreezeFrameConfig instance.

    Returns
    -------
    dict
        Feature-name -> feature-value mapping.

    Notes
    -----
    Distances are returned in metres.
    Angles are returned in radians.
    """

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if not isinstance(n_nearest_players, int) or n_nearest_players < 0:
        raise ValueError(
            "n_nearest_players must be a non-negative integer."
        )

    if distance_thresholds_m is not None:
        if any(threshold < 0 for threshold in distance_thresholds_m):
            raise ValueError(
                "All distance thresholds must be >= 0."
            )

    # -------------------------------------------------------------
    # Missing freeze-frame
    # -------------------------------------------------------------

    if (
        freeze_frame is None
        or not isinstance(freeze_frame, list)
        or shot_location is None
    ):
        return {}

    # -------------------------------------------------------------
    # Geometry setup
    # -------------------------------------------------------------

    ball = statsbomb_to_metres(shot_location, config)
    post_1, post_2, goal_center = get_goal_points(config)
    goal_interval = goal_angle_interval(ball, post_1, post_2, goal_center)
    defenders, teammates, gk = parse_freeze_frame(freeze_frame, config)
    goal_line_x_m = config.goal_x * config.pitch_length_m / config.sb_pitch_length

    features = {}

    # =============================================================
    # DISTANCE
    # =============================================================

    features["shot_distance_to_goal"] = abs(goal_line_x_m - ball[0])
    features["shot_distance_to_goal_center"] = euclidean_distance(ball, goal_center)

    # =============================================================
    # DEFENDERS
    # =============================================================

    features.update(
        nearest_distance_features(
            ball=ball,
            players=defenders,
            n_nearest=n_nearest_players,
            prefix="nearest_defender",
        )
    )

    features.update(
        within_distance_features(
            ball=ball,
            players=defenders,
            thresholds_m=distance_thresholds_m,
            prefix="defenders_within",
        )
    )

    # =============================================================
    # TRIANGLE
    # =============================================================

    if include_triangle_vars:
        defenders_triangle = players_in_triangle(
            players=defenders,
            ball=ball,
            post_1=post_1,
            post_2=post_2,
        )

        defender_triangle_distances = [
            euclidean_distance(ball, defender)
            for defender in defenders_triangle
        ]

        features["defenders_in_shot_triangle"] = len(defenders_triangle)

        features["closest_defender_in_shot_triangle"] = (
            min(defender_triangle_distances)
            if defender_triangle_distances
            else np.nan
        )

        features["avg_distance_defenders_in_triangle"] = (
            float(np.mean(defender_triangle_distances))
            if defender_triangle_distances
            else np.nan
        )

    # =============================================================
    # GOAL ANGLE
    # =============================================================
    if include_goal_angle_vars:
        total_goal_angle = goal_interval[1] - goal_interval[0]

        blocked_goal_angle = calculate_angle_coverage(
            ball=ball,
            players=defenders,
            goal_interval=goal_interval,
            goal_center=goal_center,
            radius_m=config.defender_radius_m,
        )

        blocked_goal_angle = min(blocked_goal_angle, total_goal_angle)

        open_goal_angle = max(total_goal_angle - blocked_goal_angle, 0.0)

        fraction_goal_angle_blocked = (
            blocked_goal_angle / total_goal_angle
            if total_goal_angle > 0
            else np.nan
        )

        features.update(
            {
                "open_goal_angle": open_goal_angle,
                "blocked_goal_angle": blocked_goal_angle,
                "fraction_goal_angle_blocked": fraction_goal_angle_blocked,
            }
        )

    # =============================================================
    # GOALKEEPER
    # =============================================================

    if include_goalkeeper_vars:
        if gk is None:
            features.update(
                {
                    "gk_distance_to_ball": np.nan,
                    "gk_distance_to_goal": np.nan,
                    "gk_distance_to_goal_center": np.nan,
                    "gk_distance_to_shot_centerline": np.nan,
                    "gk_lateral_position": np.nan,
                    "gk_goal_angle_coverage": np.nan,
                }
            )

        else:
            features["gk_distance_to_ball"] = euclidean_distance(gk, ball)
            features["gk_distance_to_goal"] = abs(goal_line_x_m - gk[0])
            features["gk_distance_to_goal_center"] = euclidean_distance(gk, goal_center)
            features["gk_distance_to_shot_centerline"] = distance_point_to_line(gk, ball, goal_center)
            features["gk_lateral_position"] = gk[1] - goal_center[1]
            features["gk_goal_angle_coverage"] = (
                calculate_angle_coverage(
                    ball=ball,
                    players=[gk],
                    goal_interval=goal_interval,
                    goal_center=goal_center,
                    radius_m=config.goalkeeper_radius_m,
                )
            )

    # =============================================================
    # TEAMMATES
    # =============================================================

    if include_teammates_vars:
        features.update(
            nearest_distance_features(
                ball=ball,
                players=teammates,
                n_nearest=n_nearest_players,
                prefix="nearest_teammate",
            )
        )

        features.update(
            within_distance_features(
                ball=ball,
                players=teammates,
                thresholds_m=distance_thresholds_m,
                prefix="teammates_within",
            )
        )

        if include_triangle_vars:
            teammates_triangle = players_in_triangle(
                players=teammates,
                ball=ball,
                post_1=post_1,
                post_2=post_2,
            )
            features["teammates_in_shot_triangle"] = (
                len(teammates_triangle)
            )

    return features
