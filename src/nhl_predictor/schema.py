"""Canonical data contracts for point-in-time NHL prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

REQUIRED_GAME_COLUMNS = (
    "game_id",
    "start_time_utc",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
)

OBSERVATION_COLUMNS = (
    "observation_id",
    "source_kind",
    "source_tier",
    "source_url",
    "fetched_at_utc",
    "published_at_utc",
    "available_at_utc",
    "payload_hash",
    "game_id",
    "team",
    "player_id",
)

PLAYER_STATUS_COLUMNS = (
    "player_id",
    "team",
    "status",
    "detail",
    "available_at_utc",
    "source_observation_id",
)

GAME_CUTOFF_COLUMNS = (
    "game_id",
    "forecast_kind",
    "cutoff_utc",
)

PREDICTION_COLUMNS = (
    "prediction_id",
    "game_id",
    "forecast_kind",
    "cutoff_utc",
    "snapshot_id",
    "model_version",
    "home_win_probability",
    "expected_home_goals",
    "expected_away_goals",
)

FEATURE_COLUMNS = (
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_goal_rate",
    "away_goal_rate",
    "home_goal_allowed_rate",
    "away_goal_allowed_rate",
    "home_rest_days",
    "away_rest_days",
    "rest_difference",
    "home_h2h_win_rate",
    "games_played_difference",
    "home_expected_goal_rate",
    "away_expected_goal_rate",
    "home_expected_goal_allowed_rate",
    "away_expected_goal_allowed_rate",
    "expected_goal_rate_difference",
)

# Named groups exist so each block of work can be switched off and measured.
# A feature group that does not improve out-of-sample log loss gets deleted,
# however reasonable it sounds.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "elo": ("home_elo", "away_elo", "elo_difference"),
    "scoring": (
        "home_goal_rate",
        "away_goal_rate",
        "home_goal_allowed_rate",
        "away_goal_allowed_rate",
    ),
    "rest": ("home_rest_days", "away_rest_days", "rest_difference"),
    "head_to_head": ("home_h2h_win_rate",),
    "experience": ("games_played_difference",),
    "expected_goals": (
        "home_expected_goal_rate",
        "away_expected_goal_rate",
        "home_expected_goal_allowed_rate",
        "away_expected_goal_allowed_rate",
        "expected_goal_rate_difference",
    ),
}


class SourceTier(StrEnum):
    """Confidence tier for incoming information."""

    OFFICIAL = "official"
    LICENSED = "licensed"
    SOCIAL_UNCONFIRMED = "social_unconfirmed"


class PlayerStatus(StrEnum):
    """Statuses suitable for predictive features, never free-form injury text."""

    CONFIRMED_OUT = "confirmed_out"
    DOUBTFUL = "doubtful"
    QUESTIONABLE = "questionable"
    CONFIRMED_STARTER = "confirmed_starter"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    """An immutable source record as it was available to the system."""

    observation_id: str
    source_kind: str
    source_tier: SourceTier
    source_url: str
    fetched_at_utc: str
    available_at_utc: str
    payload_hash: str
    published_at_utc: str | None = None
    game_id: str | None = None
    team: str | None = None
    player_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
