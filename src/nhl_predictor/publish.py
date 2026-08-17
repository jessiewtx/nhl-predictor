"""Build the JSON snapshot the dashboard reads.

The site is deliberately static. Predictions change once or twice a day, so a
published snapshot loads faster than a database round trip, costs nothing, and
keeps a single source of truth: whatever the pipeline last produced.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nhl_predictor.backtest import calibration_by_bucket, summarize_backtest

MAX_RECENT_GAMES = 400


def _round(value: object, digits: int = 4) -> object:
    return round(float(value), digits) if isinstance(value, int | float) else value


def build_snapshot(
    backtest: pd.DataFrame,
    games: pd.DataFrame,
    ablation: pd.DataFrame | None = None,
    upcoming: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Assemble headline metrics, recent graded predictions, and today's slate."""

    summary = summarize_backtest(backtest)
    identity = games.loc[:, ["game_id", "start_time_utc", "home_team", "away_team"]].copy()
    identity["game_id"] = identity["game_id"].astype(str)

    graded = backtest.copy()
    graded["game_id"] = graded["game_id"].astype(str)
    graded = graded.merge(identity, on="game_id", how="left")
    graded = graded.sort_values("start_time_utc", ascending=False).head(MAX_RECENT_GAMES)
    graded["predicted_home_win"] = graded["home_win_probability"] >= 0.5
    graded["correct"] = graded["predicted_home_win"] == graded["home_win"].astype(bool)

    recent = [
        {
            "game_id": row["game_id"],
            "start_time_utc": row["start_time_utc"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_win_probability": _round(row["home_win_probability"]),
            "expected_home_goals": _round(row["expected_home_goals"], 2),
            "expected_away_goals": _round(row["expected_away_goals"], 2),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "home_win": bool(row["home_win"]),
            "correct": bool(row["correct"]),
        }
        for _, row in graded.iterrows()
    ]

    snapshot: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "headline": {
            "games_evaluated": int(summary["games"]),
            "accuracy": _round(summary["accuracy"]),
            "always_home_accuracy": _round(summary["always_home_accuracy"]),
            "accuracy_lift": _round(summary["accuracy_lift_over_always_home"]),
            "log_loss": _round(summary["log_loss"]),
            "brier_score": _round(summary["brier_score"]),
            "home_goal_mae": _round(summary["home_goal_mae"], 3),
            "away_goal_mae": _round(summary["away_goal_mae"], 3),
        },
        "calibration": [
            {
                "range": bucket["range"],
                "games": bucket["games"],
                "predicted": _round(bucket["mean_probability"], 3),
                "actual": _round(bucket["actual_home_win_rate"], 3),
            }
            for bucket in calibration_by_bucket(backtest)
        ],
        "recent": recent,
        "upcoming": [],
        "ablation": [],
    }

    if ablation is not None:
        snapshot["ablation"] = [
            {
                "configuration": row["configuration"],
                "features": int(row["features"]),
                "accuracy": _round(row["accuracy"]),
                "log_loss": _round(row["log_loss"], 5),
                "log_loss_gain": None
                if pd.isna(row.get("log_loss_gain"))
                else _round(row["log_loss_gain"], 5),
            }
            for _, row in ablation.iterrows()
        ]

    if upcoming is not None and len(upcoming):
        snapshot["upcoming"] = [
            {
                "game_id": str(row["game_id"]),
                "start_time_utc": row["start_time_utc"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_probability": _round(row["home_win_probability"]),
                "expected_home_goals": _round(row["expected_home_goals"], 2),
                "expected_away_goals": _round(row["expected_away_goals"], 2),
            }
            for _, row in upcoming.iterrows()
        ]

    return snapshot


def write_snapshot(snapshot: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n")
    return output
