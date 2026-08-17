"""Walk-forward evaluation that mirrors a daily production workflow."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from nhl_predictor.cutoffs import with_forecast_cutoff
from nhl_predictor.expected_goals import EVALUATION_EXCLUDED_COLUMN
from nhl_predictor.features import make_training_frame
from nhl_predictor.predictor import NhlPredictor
from nhl_predictor.schema import FEATURE_GROUPS


def walk_forward_backtest(
    games: pd.DataFrame,
    minimum_training_games: int = 300,
    forecast_kind: str = "final",
    feature_columns: Sequence[str] | None = None,
    model_factory: Callable[[Sequence[str] | None], object] = NhlPredictor,
) -> pd.DataFrame:
    """Replay each forecast from only labels finalized before that game's cutoff."""

    frame = make_training_frame(with_forecast_cutoff(games, forecast_kind))
    predictions: list[pd.DataFrame] = []
    for cutoff_utc, test in frame.groupby("prediction_cutoff_utc", sort=True):
        train = frame[frame["result_available_at_utc"] <= cutoff_utc]
        if len(train) < minimum_training_games:
            continue

        # Games a feature model was fitted on stay in training, where they are
        # legitimate history, but are never scored. Reporting on them would
        # grade a model against data it already learned from.
        if EVALUATION_EXCLUDED_COLUMN in test.columns:
            test = test[~test[EVALUATION_EXCLUDED_COLUMN].fillna(False).astype(bool)]
        if test.empty:
            continue

        model = model_factory(feature_columns).fit(train)
        day_predictions = model.predict_features(test).loc[
            :, ["game_id", "home_win_probability", "expected_home_goals", "expected_away_goals"]
        ]
        day_predictions["game_day"] = test["game_day"].to_numpy()
        day_predictions["cutoff_utc"] = cutoff_utc
        day_predictions["forecast_kind"] = forecast_kind
        day_predictions["home_win"] = test["home_win"].to_numpy()
        day_predictions["home_score"] = test["home_score"].to_numpy()
        day_predictions["away_score"] = test["away_score"].to_numpy()
        predictions.append(day_predictions)

    if not predictions:
        raise ValueError(
            "No backtest predictions produced. Lower minimum_training_games or supply more history."
        )
    return pd.concat(predictions, ignore_index=True)


def walk_forward_home_rate(
    games: pd.DataFrame, minimum_training_games: int = 300, forecast_kind: str = "final"
) -> pd.DataFrame:
    """The honest floor: predict the home-win rate observed before each cutoff.

    Any feature work has to beat this, otherwise it is an expensive way to
    rediscover home-ice advantage.
    """

    frame = make_training_frame(with_forecast_cutoff(games, forecast_kind))
    predictions: list[pd.DataFrame] = []
    for cutoff_utc, test in frame.groupby("prediction_cutoff_utc", sort=True):
        train = frame[frame["result_available_at_utc"] <= cutoff_utc]
        if len(train) < minimum_training_games:
            continue

        if EVALUATION_EXCLUDED_COLUMN in test.columns:
            test = test[~test[EVALUATION_EXCLUDED_COLUMN].fillna(False).astype(bool)]
        if test.empty:
            continue

        home_rate = float(train["home_win"].mean())
        day_predictions = test.loc[:, ["game_id"]].copy()
        day_predictions["home_win_probability"] = home_rate
        day_predictions["expected_home_goals"] = float(train["home_score"].mean())
        day_predictions["expected_away_goals"] = float(train["away_score"].mean())
        day_predictions["cutoff_utc"] = cutoff_utc
        day_predictions["home_win"] = test["home_win"].to_numpy()
        day_predictions["home_score"] = test["home_score"].to_numpy()
        day_predictions["away_score"] = test["away_score"].to_numpy()
        predictions.append(day_predictions)

    if not predictions:
        raise ValueError("No baseline predictions produced; supply more history.")
    return pd.concat(predictions, ignore_index=True)


def run_ablation(
    games: pd.DataFrame,
    ladder: Sequence[str] = (
        "elo",
        "scoring",
        "rest",
        "head_to_head",
        "experience",
        "expected_goals",
    ),
    minimum_training_games: int = 300,
    forecast_kind: str = "final",
) -> pd.DataFrame:
    """Add one feature group at a time and report what each one actually buys.

    Log loss is the column that decides. Accuracy moves in noisy steps because
    it only cares which side of 0.5 a forecast lands on.
    """

    unknown = set(ladder) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

    rows: list[dict[str, object]] = []
    baseline = summarize_backtest(
        walk_forward_home_rate(games, minimum_training_games, forecast_kind)
    )
    rows.append({"configuration": "home_win_rate_only", "features": 0, **_headline(baseline)})

    columns: list[str] = []
    for group in ladder:
        columns.extend(FEATURE_GROUPS[group])
        summary = summarize_backtest(
            walk_forward_backtest(games, minimum_training_games, forecast_kind, columns)
        )
        rows.append(
            {
                "configuration": f"+{group}",
                "features": len(columns),
                **_headline(summary),
            }
        )

    table = pd.DataFrame(rows)
    table["log_loss_gain"] = table["log_loss"].shift(1) - table["log_loss"]
    return table


def _headline(summary: dict[str, object]) -> dict[str, object]:
    return {
        key: summary[key]
        for key in ("games", "accuracy", "log_loss", "brier_score", "home_goal_mae")
    }


def calibration_by_bucket(predictions: pd.DataFrame, buckets: int = 5) -> list[dict[str, float | int]]:
    """Compare forecast confidence to actual win frequency."""

    frame = predictions.copy()
    frame["bucket"] = pd.cut(
        frame["home_win_probability"],
        bins=np.linspace(0, 1, buckets + 1),
        include_lowest=True,
    )
    grouped = frame.groupby("bucket", observed=True)
    return [
        {
            "range": str(bucket),
            "games": len(group),
            "mean_probability": float(group["home_win_probability"].mean()),
            "actual_home_win_rate": float(group["home_win"].mean()),
        }
        for bucket, group in grouped
    ]


def summarize_backtest(predictions: pd.DataFrame) -> dict[str, object]:
    """Return proper probability metrics plus score mean absolute error."""

    probabilities = predictions["home_win_probability"].clip(1e-6, 1 - 1e-6)
    home_win = predictions["home_win"]
    accuracy = float(((probabilities >= 0.5) == home_win).mean())
    always_home_accuracy = float(home_win.mean())
    return {
        "games": len(predictions),
        "accuracy": accuracy,
        "always_home_accuracy": always_home_accuracy,
        "accuracy_lift_over_always_home": accuracy - always_home_accuracy,
        "brier_score": float(brier_score_loss(home_win, probabilities)),
        "log_loss": float(log_loss(home_win, probabilities)),
        "home_goal_mae": float(
            np.abs(predictions["expected_home_goals"] - predictions["home_score"]).mean()
        ),
        "away_goal_mae": float(
            np.abs(predictions["expected_away_goals"] - predictions["away_score"]).mean()
        ),
        "calibration_by_bucket": calibration_by_bucket(predictions),
    }
