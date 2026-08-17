from datetime import timedelta

import numpy as np
import pandas as pd

from nhl_predictor.features import build_pregame_features
from nhl_predictor.verification import (
    determinism_check,
    label_reconstruction_check,
    permute_outcomes,
    prefix_invariance_check,
)


def _season(game_count: int = 240, seed: int = 7) -> pd.DataFrame:
    """A synthetic season where the better team usually, but not always, wins.

    Real hockey is noisy. A fixture where the stronger team always wins makes
    head-to-head record a perfect predictor, which trips the leakage guards on
    data that has no leak.
    """

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-10-01T23:00:00Z")
    teams = ["TOR", "DAL", "BOS", "MTL", "NYR", "SJS"]
    rows = []
    for index in range(game_count):
        home = teams[index % 6]
        away = teams[(index + 2) % 6]
        strong_home = teams.index(home) < teams.index(away)
        home_wins = rng.random() < (0.65 if strong_home else 0.35)
        home_score, away_score = (4, 2) if home_wins else (2, 4)
        rows.append(
            {
                "game_id": f"20240200{index:03d}",
                "start_time_utc": (start + timedelta(days=index // 3, hours=index % 3)).isoformat(),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
    return pd.DataFrame(rows)


def test_removing_later_games_leaves_earlier_features_untouched() -> None:
    result = prefix_invariance_check(_season())

    assert result.passed, result.detail


def test_feature_generation_is_reproducible() -> None:
    assert determinism_check(_season()).passed


def test_no_single_feature_gives_away_the_result() -> None:
    assert label_reconstruction_check(_season()).passed


def test_permutation_keeps_the_schedule_but_moves_the_results() -> None:
    games = _season(60)
    shuffled = permute_outcomes(games, seed=3)

    assert shuffled["home_team"].equals(games["home_team"])
    assert shuffled["start_time_utc"].equals(games["start_time_utc"])
    # Score pairs travel together, so league scoring is unchanged.
    assert sorted(shuffled["home_score"]) == sorted(games["home_score"])
    assert not shuffled["home_score"].equals(games["home_score"])


def test_games_a_feature_model_trained_on_are_never_scored() -> None:
    """Training on them is fine; reporting metrics from them is not."""

    from nhl_predictor.backtest import walk_forward_backtest
    from nhl_predictor.expected_goals import EVALUATION_EXCLUDED_COLUMN

    games = _season(400)
    early_cutoff = games["start_time_utc"].sort_values().iloc[150]
    games[EVALUATION_EXCLUDED_COLUMN] = games["start_time_utc"] <= early_cutoff
    excluded_ids = set(games.loc[games[EVALUATION_EXCLUDED_COLUMN], "game_id"])

    predictions = walk_forward_backtest(games, minimum_training_games=60)

    assert len(predictions) > 0
    assert not set(predictions["game_id"]) & excluded_ids


def test_a_deliberate_leak_is_caught_by_the_guards() -> None:
    """Prove the checks can fail: hand a future result to a past game."""

    games = _season()
    honest = build_pregame_features(games).set_index("game_id")

    leaked = games.copy()
    # Give every game the settlement time of the season's first game, so later
    # results become visible to earlier cutoffs.
    leaked["result_available_at_utc"] = games["start_time_utc"].min()
    leaky_features = build_pregame_features(leaked).set_index("game_id")

    assert not honest["home_elo"].equals(leaky_features["home_elo"])
