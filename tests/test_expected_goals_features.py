import pandas as pd

from nhl_predictor.expected_goals import attach_team_expected_goals
from nhl_predictor.features import LEAGUE_AVERAGE_EXPECTED_GOALS, build_pregame_features


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
                "start_time_utc": "2025-10-05T23:00:00Z",
                "home_team": "TOR",
                "away_team": "BOS",
                "home_score": 1,
                "away_score": 2,
            },
        ]
    )


def test_features_rest_at_the_prior_when_no_shot_data_exists() -> None:
    features = build_pregame_features(_games()).set_index("game_id")

    assert features["home_expected_goal_rate"].eq(LEAGUE_AVERAGE_EXPECTED_GOALS).all()
    assert features["expected_goal_rate_difference"].eq(0.0).all()


def test_chance_quality_from_an_earlier_game_reaches_the_next_one() -> None:
    totals = pd.DataFrame(
        [
            {"game_id": "1", "is_home_shooter": True, "expected_goals_for": 5.0},
            {"game_id": "1", "is_home_shooter": False, "expected_goals_for": 1.0},
        ]
    )
    games = attach_team_expected_goals(_games(), totals)

    features = build_pregame_features(games).set_index("game_id")

    # Toronto out-chanced Dallas badly in game 1, so game 2 sees a raised rate.
    assert features.loc["2", "home_expected_goal_rate"] > LEAGUE_AVERAGE_EXPECTED_GOALS
    assert features.loc["2", "expected_goal_rate_difference"] > 0
    # Boston has no history, so it stays neutral rather than borrowing Dallas's.
    assert features.loc["2", "away_expected_goal_rate"] == LEAGUE_AVERAGE_EXPECTED_GOALS
