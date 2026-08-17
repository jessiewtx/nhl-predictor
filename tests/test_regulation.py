from datetime import timedelta

import pandas as pd
import pytest

from nhl_predictor.features import make_training_frame
from nhl_predictor.regulation import RegulationPredictor, label_regulation_outcome


def _season(game_count: int = 300) -> pd.DataFrame:
    start = pd.Timestamp("2024-10-01T23:00:00Z")
    teams = ["TOR", "DAL", "BOS", "MTL"]
    rows = []
    for index in range(game_count):
        home = teams[index % 4]
        away = teams[(index + 1) % 4]
        # Every fifth game reaches overtime, near the real league rate.
        beyond = index % 5 == 0
        home_score, away_score = (4, 3) if index % 3 else (2, 3)
        rows.append(
            {
                "game_id": f"20240200{index:03d}",
                "start_time_utc": (start + timedelta(days=index // 4, hours=index % 4)).isoformat(),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "last_period_type": "OT" if beyond else "REG",
            }
        )
    return pd.DataFrame(rows)


def test_overtime_games_are_labeled_tied_after_regulation() -> None:
    games = _season(10)

    outcomes = label_regulation_outcome(games)

    # Game 0 went to overtime, so its regulation state is a tie despite a winner.
    assert outcomes.iloc[0] == "tie"
    assert set(outcomes) <= {"home", "away", "tie"}


def test_missing_period_type_fails_loudly_rather_than_guessing() -> None:
    games = _season(10).drop(columns=["last_period_type"])

    with pytest.raises(ValueError, match="last_period_type"):
        label_regulation_outcome(games)


def test_predictions_decompose_into_regulation_and_overtime() -> None:
    frame = make_training_frame(_season())
    model = RegulationPredictor().fit(frame)

    predictions = model.predict_features(frame)

    assert predictions["home_win_probability"].between(0, 1).all()
    assert predictions["overtime_probability"].between(0, 1).all()
    # A home win is a regulation win plus the share of ties won in overtime.
    assert (
        predictions["home_win_probability"] >= predictions["regulation_home_win_probability"] - 1e-9
    ).all()
