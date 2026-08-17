"""ESPN player-status feed, the source behind ESPN's fantasy player pages.

Why this exists: the NHL publishes no centralized, machine-readable injury
report, and most club injury pages render client side, so our official capture
returns navigation markup instead of players. ESPN maintains a per-player
status record that carries its own ``date`` stamp, which lets us record when a
status was published rather than only when we happened to fetch it.

Two cautions are wired into this module rather than left to memory:

* ESPN's status answers "is this player available at all", not "is this player
  out for tonight". An offseason record can read ``Out`` for months.
* ``details.returnDate`` is frequently a generic placeholder shared by every
  open record, so it is carried through unmodified and never trusted as a
  projection.

The feed is a third-party aggregator, so it is captured as ``LICENSED`` rather
than ``OFFICIAL``, and its statuses are treated as candidate labels that a
reviewer confirms.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from nhl_predictor.ledger import ObservationStore
from nhl_predictor.schema import PlayerStatus, SourceTier

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
TEAMS_URL = f"{ESPN_BASE}/teams?limit=40"
TEAM_INJURIES_URL = f"{ESPN_BASE}/teams/{{team_id}}/injuries?limit=100"
REQUEST_HEADERS = {"User-Agent": "nhl-predictor/0.1 (research; contact via repository)"}
REQUEST_DELAY_SECONDS = 0.05

# ESPN's vocabulary, mapped onto availability. A suspension is not an injury,
# but for lineup purposes it is the same fact: the player cannot dress.
STATUS_MAP = {
    "out": PlayerStatus.CONFIRMED_OUT,
    "injured reserve": PlayerStatus.CONFIRMED_OUT,
    "suspension": PlayerStatus.CONFIRMED_OUT,
    "doubtful": PlayerStatus.DOUBTFUL,
    "day-to-day": PlayerStatus.QUESTIONABLE,
    "day to day": PlayerStatus.QUESTIONABLE,
    "questionable": PlayerStatus.QUESTIONABLE,
    "game time decision": PlayerStatus.QUESTIONABLE,
    "game-time decision": PlayerStatus.QUESTIONABLE,
}

INJURY_COLUMNS = (
    "injury_id",
    "team",
    "player_name",
    "player_id",
    "espn_status",
    "status",
    "published_at_utc",
    "injury_type",
    "injury_detail",
    "reported_return_date",
    "short_comment",
    "long_comment",
    "source_url",
)


def map_status(espn_status: str | None) -> PlayerStatus:
    """Map an ESPN status string onto our availability enum.

    Unrecognized values become ``UNKNOWN`` instead of a guess, so a new ESPN
    label surfaces as missing information rather than a silent wrong answer.
    """

    if not espn_status:
        return PlayerStatus.UNKNOWN
    return STATUS_MAP.get(espn_status.strip().lower(), PlayerStatus.UNKNOWN)


def _identifier(reference: str) -> str:
    return reference.rstrip("/").split("/")[-1].split("?")[0]


def _get(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url.replace("http://", "https://"), timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_team_index(session: requests.Session | None = None) -> dict[str, str]:
    """Return ESPN team id to team abbreviation."""

    client = session or requests.Session()
    client.headers.update(REQUEST_HEADERS)
    index: dict[str, str] = {}
    for item in _get(client, TEAMS_URL).get("items", []):
        team = _get(client, item["$ref"])
        index[str(team["id"])] = team.get("abbreviation", team.get("shortDisplayName", ""))
        time.sleep(REQUEST_DELAY_SECONDS)
    return index


def fetch_team_injuries(
    team_id: str, session: requests.Session | None = None
) -> list[dict[str, Any]]:
    """Fetch and dereference every current injury record for one team."""

    client = session or requests.Session()
    client.headers.update(REQUEST_HEADERS)
    index = _get(client, TEAM_INJURIES_URL.format(team_id=team_id))
    records = []
    for item in index.get("items", []):
        records.append(_get(client, item["$ref"]))
        time.sleep(REQUEST_DELAY_SECONDS)
    return records


def normalize_injury_record(record: dict[str, Any], team: str, player_name: str) -> dict[str, Any]:
    """Flatten one ESPN injury record onto our canonical fields."""

    details = record.get("details") or {}
    return {
        "injury_id": str(record.get("id", "")),
        "team": team,
        "player_name": player_name,
        "player_id": _identifier(record.get("athlete", {}).get("$ref", "")),
        "espn_status": record.get("status"),
        "status": str(map_status(record.get("status"))),
        "published_at_utc": record.get("date"),
        "injury_type": details.get("type"),
        "injury_detail": details.get("detail"),
        "reported_return_date": details.get("returnDate"),
        "short_comment": record.get("shortComment"),
        "long_comment": record.get("longComment"),
        "source_url": record.get("$ref", "").replace("http://", "https://"),
    }


def collect_league_injuries(
    store: ObservationStore | None = None, session: requests.Session | None = None
) -> pd.DataFrame:
    """Capture every team's current player statuses, one immutable record each.

    ``published_at_utc`` is ESPN's own stamp for the status. ``available_at_utc``
    stays at fetch time because that is the earliest moment this system could
    have known it.
    """

    client = session or requests.Session()
    client.headers.update(REQUEST_HEADERS)
    rows: list[dict[str, Any]] = []

    for team_id, abbreviation in fetch_team_index(client).items():
        for record in fetch_team_injuries(team_id, client):
            athlete_reference = record.get("athlete", {}).get("$ref")
            player_name = ""
            if athlete_reference:
                athlete = _get(client, athlete_reference)
                player_name = athlete.get("displayName", "")
                time.sleep(REQUEST_DELAY_SECONDS)
            row = normalize_injury_record(record, abbreviation, player_name)
            rows.append(row)
            if store is not None:
                store.capture(
                    record,
                    source_kind="espn_player_status",
                    source_url=row["source_url"],
                    source_tier=SourceTier.LICENSED,
                    published_at_utc=row["published_at_utc"],
                    team=abbreviation,
                    player_id=row["player_id"] or None,
                )

    if not rows:
        return pd.DataFrame(columns=list(INJURY_COLUMNS))
    return pd.DataFrame(rows, columns=list(INJURY_COLUMNS))
