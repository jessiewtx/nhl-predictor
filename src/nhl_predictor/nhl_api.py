"""Thin client for the NHL's public schedule endpoint.

This API is an upstream convenience, not an immutable historical archive. Save
the normalized CSV returned by this module so every experiment is reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from nhl_predictor.ledger import ObservationStore

SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{game_date}"
GAME_CENTER_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/{feed}"
OFFICIAL_INJURY_REPORT_URLS = (
    "https://www.nhl.com/ducks/team/injury-report",
    "https://www.nhl.com/bruins/team/injury-report",
    "https://www.nhl.com/sabres/team/injury-report",
    "https://www.nhl.com/flames/team/injury-report",
    "https://www.nhl.com/hurricanes/team/injury-report",
    "https://www.nhl.com/blackhawks/team/injury-report",
    "https://www.nhl.com/avalanche/team/injury-report",
    "https://www.nhl.com/bluejackets/team/injury-report",
    "https://www.nhl.com/stars/team/injury-report",
    "https://www.nhl.com/redwings/team/injury-report",
    "https://www.nhl.com/oilers/team/injury-report",
    "https://www.nhl.com/panthers/team/injury-report",
    "https://www.nhl.com/kings/team/injury-report",
    "https://www.nhl.com/wild/team/injury-report",
    "https://www.nhl.com/canadiens/team/injury-report",
    "https://www.nhl.com/predators/team/injury-report",
    "https://www.nhl.com/devils/team/injury-report",
    "https://www.nhl.com/islanders/team/injury-report",
    "https://www.nhl.com/rangers/team/injury-report",
    "https://www.nhl.com/senators/team/injury-report",
    "https://www.nhl.com/flyers/team/injury-report",
    "https://www.nhl.com/penguins/team/injury-report",
    "https://www.nhl.com/sharks/team/injury-report",
    "https://www.nhl.com/kraken/team/injury-report",
    "https://www.nhl.com/blues/team/injury-report",
    "https://www.nhl.com/lightning/team/injury-report",
    "https://www.nhl.com/mapleleafs/team/injury-report",
    "https://www.nhl.com/utah/team/injury-report",
    "https://www.nhl.com/canucks/team/injury-report",
    "https://www.nhl.com/goldenknights/team/injury-report",
    "https://www.nhl.com/capitals/team/injury-report",
    "https://www.nhl.com/jets/team/injury-report",
)


def normalize_schedule(payload: dict, regular_season_only: bool = True) -> pd.DataFrame:
    """Normalize a schedule response to the project's canonical game schema."""

    rows: list[dict[str, object]] = []
    for game_week in payload.get("gameWeek", []):
        for game in game_week.get("games", []):
            if regular_season_only and game.get("gameType") != 2:
                continue
            home_score = game.get("homeTeam", {}).get("score")
            away_score = game.get("awayTeam", {}).get("score")
            rows.append(
                {
                    "game_id": str(game["id"]),
                    "season": game.get("season"),
                    "game_date": game_week.get("date"),
                    "start_time_utc": game["startTimeUTC"],
                    "home_team": game["homeTeam"]["abbrev"],
                    "away_team": game["awayTeam"]["abbrev"],
                    "home_score": home_score,
                    "away_score": away_score,
                    "game_state": game.get("gameState"),
                    # REG, OT, or SO. Roughly a fifth of games are decided
                    # after regulation, where the winner is close to a coin
                    # flip, so the three outcomes are worth modeling apart.
                    "last_period_type": (game.get("gameOutcome") or {}).get("lastPeriodType"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "game_date",
                "start_time_utc",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "game_state",
                "last_period_type",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("game_id")


def build_session() -> requests.Session:
    """A session that waits out rate limits instead of dropping the request.

    Bulk collection over thousands of games will hit 429s. Without retries the
    caller silently loses those games, leaving gaps that are easy to mistake
    for missing data rather than a failed fetch.
    """

    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_schedule_day(game_date: date, session: requests.Session | None = None) -> pd.DataFrame:
    client = session or build_session()
    response = client.get(SCHEDULE_URL.format(game_date=game_date.isoformat()), timeout=30)
    response.raise_for_status()
    return normalize_schedule(response.json())


def fetch_game_center_feed(
    game_id: str, feed: str, session: requests.Session | None = None
) -> dict:
    """Fetch an official NHL Game Center box-score or play-by-play payload."""

    if feed not in {"boxscore", "play-by-play"}:
        raise ValueError("feed must be 'boxscore' or 'play-by-play'")
    client = session or build_session()
    response = client.get(GAME_CENTER_URL.format(game_id=game_id, feed=feed), timeout=30)
    response.raise_for_status()
    return response.json()


def capture_official_observations(
    game_date: date,
    store: ObservationStore,
    *,
    include_injury_reports: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Capture raw official data now, preserving when it became available.

    Club injury reports are intentionally saved as immutable HTML rather than
    parsed with a brittle universal scraper. Player statuses are only promoted
    after a source-specific parser verifies the team and timestamp.
    """

    client = session or build_session()
    schedule_url = SCHEDULE_URL.format(game_date=game_date.isoformat())
    response = client.get(schedule_url, timeout=30)
    response.raise_for_status()
    schedule_payload = response.json()
    store.capture(schedule_payload, source_kind="nhl_schedule", source_url=schedule_url)
    games = normalize_schedule(schedule_payload)
    games = games[games["game_date"] == game_date.isoformat()].copy()

    for _, game in games.dropna(subset=["home_score", "away_score"]).iterrows():
        for feed in ("boxscore", "play-by-play"):
            source_url = GAME_CENTER_URL.format(game_id=game["game_id"], feed=feed)
            feed_response = client.get(source_url, timeout=30)
            feed_response.raise_for_status()
            store.capture(
                feed_response.json(),
                source_kind=f"nhl_gamecenter_{feed.replace('-', '_')}",
                source_url=source_url,
                game_id=game["game_id"],
            )

    if include_injury_reports:
        for injury_url in OFFICIAL_INJURY_REPORT_URLS:
            injury_response = client.get(injury_url, timeout=30)
            injury_response.raise_for_status()
            store.capture(
                {"html": injury_response.text},
                source_kind="nhl_club_injury_report",
                source_url=injury_url,
                team=injury_url.split("/")[3],
            )
    return games.reset_index(drop=True)


def date_range(start_date: date, end_date: date) -> Iterable[date]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def schedule_request_dates(start_date: date, end_date: date) -> Iterable[date]:
    """Yield one request date per seven-day schedule response."""

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=7)


def collect_schedule(start_date: date, end_date: date) -> pd.DataFrame:
    """Collect completed games using the schedule endpoint's weekly responses."""

    days = [fetch_schedule_day(day) for day in schedule_request_dates(start_date, end_date)]
    if not days:
        return pd.DataFrame()
    games = pd.concat(days, ignore_index=True).drop_duplicates("game_id")
    completed = games.dropna(subset=["home_score", "away_score"]).copy()
    game_dates = pd.to_datetime(completed["game_date"]).dt.date
    completed = completed[(game_dates >= start_date) & (game_dates <= end_date)]
    return completed.sort_values("start_time_utc").reset_index(drop=True)
