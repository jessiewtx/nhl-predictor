from nhl_predictor.nhl_api import normalize_schedule


def test_normalize_schedule_keeps_regular_season_games() -> None:
    payload = {
        "gameWeek": [
            {
                "date": "2025-10-07",
                "games": [
                    {
                        "id": 2025020001,
                        "gameType": 2,
                        "startTimeUTC": "2025-10-07T23:00:00Z",
                        "gameState": "OFF",
                        "gameOutcome": {"lastPeriodType": "OT"},
                        "homeTeam": {"abbrev": "TOR", "score": 4},
                        "awayTeam": {"abbrev": "DAL", "score": 3},
                    },
                    {
                        "id": 2025010001,
                        "gameType": 1,
                        "startTimeUTC": "2025-09-22T23:00:00Z",
                        "homeTeam": {"abbrev": "TOR", "score": 2},
                        "awayTeam": {"abbrev": "MTL", "score": 1},
                    },
                ]
            }
        ]
    }

    games = normalize_schedule(payload)

    assert len(games) == 1
    assert games.iloc[0].to_dict()["game_id"] == "2025020001"
    assert games.iloc[0].to_dict()["home_team"] == "TOR"
    assert games.iloc[0].to_dict()["game_date"] == "2025-10-07"
    # Needed to separate regulation wins from near-coin-flip overtime results.
    assert games.iloc[0].to_dict()["last_period_type"] == "OT"
