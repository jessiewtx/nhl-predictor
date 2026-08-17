from datetime import timedelta

import pandas as pd

from nhl_predictor.backtest import summarize_backtest, walk_forward_backtest
from nhl_predictor.predictor import NhlPredictor


def _season(game_count: int = 42) -> pd.DataFrame:
    start = pd.Timestamp("2024-10-01T23:00:00Z")
    rows = []
    for index in range(game_count):
        home, away = ("TOR", "DAL") if index % 2 == 0 else ("DAL", "TOR")
        # Alternating results ensures both outcome labels appear in every long window.
        home_score, away_score = (4, 2) if index % 3 else (2, 4)
        rows.append(
            {
                "game_id": str(index),
                "start_time_utc": (start + timedelta(days=index)).isoformat(),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_backtest_only_predicts_after_training_window() -> None:
    predictions = walk_forward_backtest(_season(), minimum_training_games=30)
    summary = summarize_backtest(predictions)

    assert len(predictions) == 12
    assert predictions["home_win_probability"].between(0, 1).all()
    assert summary["games"] == 12
    assert summary["always_home_accuracy"] == predictions["home_win"].mean()
    assert 0 <= summary["brier_score"] <= 1


def test_predictor_scores_future_game_without_a_future_result() -> None:
    history = _season()
    schedule = pd.DataFrame(
        [
            {
                "game_id": "future",
                "start_time_utc": "2024-11-13T23:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": pd.NA,
                "away_score": pd.NA,
            }
        ]
    )

    predictions = NhlPredictor().predict_games(history, schedule)

    assert predictions["game_id"].tolist() == ["future"]
    assert predictions["home_win_probability"].between(0, 1).all()
    assert (predictions[["expected_home_goals", "expected_away_goals"]] > 0).all().all()
