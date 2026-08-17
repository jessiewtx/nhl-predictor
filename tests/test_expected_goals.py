import numpy as np
import pandas as pd
import pytest

from nhl_predictor.expected_goals import (
    ExpectedGoalsModel,
    prepare_shots,
    team_game_expected_goals,
)


def _shots(count: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = rng.uniform(5, 70, count)
    # Closer attempts score more often, which is the signal xG must recover.
    goal_probability = np.clip(0.45 - 0.006 * distance, 0.01, 0.6)
    return pd.DataFrame(
        {
            "game_id": [str(i // 100) for i in range(count)],
            "event_type": rng.choice(
                ["shot-on-goal", "missed-shot", "blocked-shot"], count, p=[0.6, 0.25, 0.15]
            ),
            "shot_type": rng.choice(["wrist", "slap", "backhand", None], count),
            "distance": distance,
            "angle": rng.uniform(0, 80, count),
            "skaters_for": rng.choice([5, 5, 5, 4], count),
            "skaters_against": 5,
            "is_empty_net": False,
            "is_home_shooter": rng.choice([True, False], count),
            "is_goal": rng.random(count) < goal_probability,
        }
    )


def test_blocked_attempts_are_excluded_and_strength_is_derived() -> None:
    prepared = prepare_shots(_shots(500))

    assert "blocked-shot" not in set(prepared["event_type"])
    assert set(prepared["event_strength"]) <= {"even", "power_play", "shorthanded", "empty_net"}
    # A missing shot type becomes an explicit category rather than a dropped row.
    assert "unknown" in set(prepared["shot_type"])


def test_model_beats_the_base_rate_on_held_out_shots() -> None:
    model = ExpectedGoalsModel().fit(_shots(4_000, seed=1))
    report = model.evaluate(_shots(2_000, seed=2))

    assert report.roc_auc > 0.6
    assert report.log_loss < 0.6
    assert report.evaluation_shots > 0


def test_scoring_requires_a_fitted_model() -> None:
    with pytest.raises(RuntimeError):
        ExpectedGoalsModel().predict(_shots(100))


def test_shot_level_expected_goals_aggregate_to_two_rows_per_game() -> None:
    shots = _shots(1_500, seed=3)
    model = ExpectedGoalsModel().fit(_shots(4_000, seed=4))
    totals = team_game_expected_goals(shots, model.predict(shots))

    assert set(totals.columns) == {
        "game_id",
        "is_home_shooter",
        "expected_goals_for",
        "attempts",
        "goals",
    }
    assert totals.groupby("game_id").size().max() <= 2
    assert (totals["expected_goals_for"] > 0).all()
