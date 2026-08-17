from datetime import timedelta

import pandas as pd

from nhl_predictor.ledger import ObservationStore
from nhl_predictor.workflow import generate_forecasts, material_prediction_changes


def _history() -> pd.DataFrame:
    start = pd.Timestamp("2024-10-01T23:00:00Z")
    rows = []
    for index in range(35):
        home, away = ("TOR", "DAL") if index % 2 == 0 else ("DAL", "TOR")
        rows.append(
            {
                "game_id": str(index),
                "start_time_utc": (start + timedelta(days=index)).isoformat(),
                "home_team": home,
                "away_team": away,
                "home_score": 4 if index % 3 else 2,
                "away_score": 2 if index % 3 else 4,
            }
        )
    return pd.DataFrame(rows)


def test_forecasts_are_recorded_with_snapshot_provenance(tmp_path) -> None:
    schedule = pd.DataFrame(
        [
            {
                "game_id": "future",
                "start_time_utc": "2024-11-07T23:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": pd.NA,
                "away_score": pd.NA,
            }
        ]
    )
    forecast = generate_forecasts(
        _history(), schedule, ObservationStore(tmp_path / "ledger"), forecast_kind="final"
    )

    assert forecast.iloc[0]["game_id"] == "future"
    assert forecast.iloc[0]["snapshot_id"]
    assert forecast.iloc[0]["model_version"] == "logistic-poisson-cutoff-v1"


def test_material_changes_catches_winner_flip() -> None:
    previous = pd.DataFrame([{"game_id": "1", "home_win_probability": 0.48}])
    current = pd.DataFrame([{"game_id": "1", "home_win_probability": 0.55}])

    changes = material_prediction_changes(previous, current)

    assert changes.iloc[0]["winner_changed"]
