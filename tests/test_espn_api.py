from nhl_predictor.espn_api import map_status, normalize_injury_record
from nhl_predictor.schema import PlayerStatus

RECORD = {
    "id": "592725",
    "status": "Injured Reserve",
    "date": "2026-07-01T22:34Z",
    "shortComment": "MacEwen (knee) signed a two-year contract with the Maple Leafs on Wednesday.",
    "longComment": "MacEwen has been sidelined since mid-November due to an ACL injury.",
    "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/seasons/2026/athletes/4063569?lang=en"},
    "details": {"type": "Knee", "detail": "Surgery", "returnDate": "2026-09-15"},
    "$ref": "http://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/seasons/2026/athletes/4063569/injuries/592725?lang=en",
}


def test_unavailability_statuses_collapse_to_confirmed_out() -> None:
    assert map_status("Out") is PlayerStatus.CONFIRMED_OUT
    assert map_status("Injured Reserve") is PlayerStatus.CONFIRMED_OUT
    # A suspension is not an injury, but the player still cannot dress.
    assert map_status("Suspension") is PlayerStatus.CONFIRMED_OUT


def test_uncertain_statuses_never_become_confirmations() -> None:
    assert map_status("Day-To-Day") is PlayerStatus.QUESTIONABLE
    assert map_status("Game Time Decision") is PlayerStatus.QUESTIONABLE
    assert map_status("Doubtful") is PlayerStatus.DOUBTFUL


def test_unrecognized_status_becomes_unknown_rather_than_a_guess() -> None:
    assert map_status("Probably Fine") is PlayerStatus.UNKNOWN
    assert map_status(None) is PlayerStatus.UNKNOWN
    assert map_status("") is PlayerStatus.UNKNOWN


def test_normalization_keeps_espn_publication_time_and_player_id() -> None:
    row = normalize_injury_record(RECORD, team="TOR", player_name="Zack MacEwen")

    assert row["player_id"] == "4063569"
    assert row["status"] == "confirmed_out"
    assert row["espn_status"] == "Injured Reserve"
    # ESPN's own stamp, not our fetch time, is what dates the claim.
    assert row["published_at_utc"] == "2026-07-01T22:34Z"
    assert row["reported_return_date"] == "2026-09-15"
    assert row["source_url"].startswith("https://")
