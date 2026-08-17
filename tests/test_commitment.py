import json

import pandas as pd

from nhl_predictor.commitment import commit_predictions, digest_predictions, verify_chain


def _predictions(probability: float = 0.61) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "1",
                "home_win_probability": probability,
                "expected_home_goals": 3.2,
                "expected_away_goals": 2.8,
            },
            {
                "game_id": "2",
                "home_win_probability": 0.44,
                "expected_home_goals": 2.7,
                "expected_away_goals": 3.1,
            },
        ]
    )


def _commit(tmp_path, predictions, kind="final"):
    return commit_predictions(
        predictions,
        forecast_kind=kind,
        cutoff_utc="2026-10-07T22:30:00+00:00",
        model_version="test-v1",
        log_path=tmp_path / "commitments.jsonl",
    )


def test_row_order_does_not_change_the_digest() -> None:
    forward = _predictions()
    reversed_rows = forward.iloc[::-1].reset_index(drop=True)

    assert digest_predictions(forward) == digest_predictions(reversed_rows)


def test_changing_a_probability_changes_the_digest() -> None:
    assert digest_predictions(_predictions(0.61)) != digest_predictions(_predictions(0.62))


def test_chain_verifies_across_multiple_commitments(tmp_path) -> None:
    first = _commit(tmp_path, _predictions(), kind="morning")
    second = _commit(tmp_path, _predictions(0.58))

    assert second.previous_hash == first.entry_hash
    valid, detail = verify_chain(tmp_path / "commitments.jsonl")
    assert valid, detail


def test_editing_a_recorded_prediction_breaks_the_chain(tmp_path) -> None:
    _commit(tmp_path, _predictions())
    _commit(tmp_path, _predictions(0.58))
    log = tmp_path / "commitments.jsonl"

    lines = log.read_text().splitlines()
    tampered = json.loads(lines[0])
    # Rewrite history the way someone would if a forecast aged badly.
    tampered["predictions_digest"] = "0" * 64
    log.write_text("\n".join([json.dumps(tampered, sort_keys=True), lines[1]]) + "\n")

    valid, detail = verify_chain(log)

    assert not valid
    assert "modified" in detail or "follow" in detail
