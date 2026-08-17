import pandas as pd

from nhl_predictor.ledger import ObservationStore


def test_snapshot_excludes_observations_unavailable_at_cutoff(tmp_path) -> None:
    store = ObservationStore(tmp_path / "ledger")
    first = store.capture(
        {"status": "confirmed"},
        source_kind="official_status",
        source_url="https://example.test/first",
        fetched_at_utc="2025-12-15T15:00:00Z",
        available_at_utc="2025-12-15T15:00:00Z",
    )
    store.capture(
        {"status": "late_update"},
        source_kind="official_status",
        source_url="https://example.test/second",
        fetched_at_utc="2025-12-15T19:00:00Z",
        available_at_utc="2025-12-15T19:00:00Z",
    )

    snapshot = store.materialize_snapshot("2025-12-15T18:00:00Z")

    assert snapshot.observation_ids == (first.observation_id,)
    assert snapshot.manifest_path.exists()


def test_prediction_ledger_records_snapshot_and_model_version(tmp_path) -> None:
    store = ObservationStore(tmp_path / "ledger")
    snapshot = store.materialize_snapshot("2025-12-15T18:00:00Z")
    recorded = store.record_predictions(
        pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "home_win_probability": 0.61,
                    "expected_home_goals": 3.2,
                    "expected_away_goals": 2.8,
                }
            ]
        ),
        snapshot_id=snapshot.snapshot_id,
        model_version="test-v1",
        forecast_kind="final",
        cutoff_utc="2025-12-15T18:00:00Z",
    )

    assert recorded.iloc[0]["snapshot_id"] == snapshot.snapshot_id
    assert recorded.iloc[0]["model_version"] == "test-v1"
