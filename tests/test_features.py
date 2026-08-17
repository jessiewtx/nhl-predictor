import pandas as pd
import pytest

from nhl_predictor.features import build_pregame_features, make_training_frame


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "1",
                "start_time_utc": "2025-10-01T23:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": 4,
                "away_score": 2,
            },
            {
                "game_id": "2",
                "start_time_utc": "2025-10-01T23:30:00Z",
                "home_team": "MTL",
                "away_team": "BOS",
                "home_score": 3,
                "away_score": 2,
            },
            {
                "game_id": "3",
                "start_time_utc": "2025-10-02T23:00:00Z",
                "home_team": "DAL",
                "away_team": "TOR",
                "home_score": 3,
                "away_score": 2,
            },
        ]
    )


def test_same_day_results_do_not_leak_into_other_pregame_features() -> None:
    features = build_pregame_features(_games()).set_index("game_id")

    assert features.loc["1", "home_elo"] == pytest.approx(1500)
    assert features.loc["2", "home_elo"] == pytest.approx(1500)
    assert features.loc["2", "home_goal_rate"] == pytest.approx(3.1)
    # The next day's Dallas game can use Toronto's completed prior-day result.
    assert features.loc["3", "away_elo"] > 1500
    assert features.loc["3", "home_elo"] < 1500


def test_training_targets_are_derived_after_feature_generation() -> None:
    training = make_training_frame(_games()).set_index("game_id")

    assert training.loc["1", "home_win"] == 1
    assert training.loc["3", "home_win"] == 1


def test_new_season_resets_short_term_stats_and_regresses_elo() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "old",
                "season": 20242025,
                "start_time_utc": "2024-10-01T23:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": 5,
                "away_score": 1,
            },
            {
                "game_id": "new",
                "season": 20252026,
                "start_time_utc": "2025-10-01T23:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": 3,
                "away_score": 2,
            },
        ]
    )

    features = build_pregame_features(games).set_index("game_id")

    assert 1500 < features.loc["new", "home_elo"] < 1510
    assert features.loc["new", "home_goal_rate"] == pytest.approx(3.1)


def test_same_day_final_score_is_used_only_after_its_availability_time() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "early",
                "start_time_utc": "2025-12-15T12:00:00Z",
                "result_available_at_utc": "2025-12-15T15:00:00Z",
                "home_team": "TOR",
                "away_team": "DAL",
                "home_score": 4,
                "away_score": 1,
            },
            {
                "game_id": "late",
                "start_time_utc": "2025-12-15T19:00:00Z",
                "prediction_cutoff_utc": "2025-12-15T18:30:00Z",
                "home_team": "DAL",
                "away_team": "TOR",
                "home_score": 2,
                "away_score": 3,
            },
        ]
    )

    features = build_pregame_features(games).set_index("game_id")

    assert features.loc["late", "away_elo"] > 1500
    assert features.loc["late", "home_elo"] < 1500
