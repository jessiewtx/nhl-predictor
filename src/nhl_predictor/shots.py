"""Shot-level features from official play-by-play, the base for expected goals.

Goals are a noisy record of how a team played: a bad team can win 2-1 on two
deflections. Shot quality is far more stable, so an expected-goals model built
from shot location, type, and strength state gives the predictor a much steadier
signal than goal counts.

Everything here derives from the NHL's own play-by-play feed, so the resulting
expected-goals model carries the same provenance and point-in-time discipline as
the rest of the pipeline, with no third-party dependency.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SHOT_EVENTS = frozenset({"goal", "shot-on-goal", "missed-shot", "blocked-shot"})
# Distance from center ice to the goal line, in NHL rink coordinates.
GOAL_LINE_X = 89.0

SHOT_COLUMNS = (
    "game_id",
    "event_id",
    "period",
    "period_type",
    "time_in_period",
    "team_id",
    "is_home_shooter",
    "shooter_id",
    "goalie_id",
    "event_type",
    "shot_type",
    "zone_code",
    "x_coord",
    "y_coord",
    "distance",
    "angle",
    "skaters_for",
    "skaters_against",
    "is_even_strength",
    "is_empty_net",
    "is_goal",
)


def _shot_geometry(x: float | None, y: float | None) -> tuple[float | None, float | None]:
    """Distance and angle to the attacking net.

    Coordinates are mirrored onto one half of the rink because teams switch
    ends between periods. The absolute value removes that bookkeeping without
    changing the geometry of the shot.
    """

    if x is None or y is None:
        return None, None
    dx = GOAL_LINE_X - abs(float(x))
    dy = float(y)
    distance = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(abs(dy), dx)) if (dx or dy) else 0.0
    return distance, angle


def _strength(situation_code: str | None, is_home_shooter: bool) -> tuple[int | None, int | None, bool]:
    """Read skater counts and whether the defending net is empty.

    ``situationCode`` packs four digits: away goalie, away skaters, home
    skaters, home goalie.
    """

    if not situation_code or len(situation_code) != 4 or not situation_code.isdigit():
        return None, None, False
    away_goalie, away_skaters, home_skaters, home_goalie = (int(d) for d in situation_code)
    if is_home_shooter:
        return home_skaters, away_skaters, away_goalie == 0
    return away_skaters, home_skaters, home_goalie == 0


def extract_shots(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten one game's play-by-play into one row per shot attempt."""

    game_id = str(payload.get("id", ""))
    home_team_id = payload.get("homeTeam", {}).get("id")
    rows: list[dict[str, Any]] = []

    for play in payload.get("plays", []):
        event_type = play.get("typeDescKey")
        if event_type not in SHOT_EVENTS:
            continue
        details = play.get("details") or {}
        team_id = details.get("eventOwnerTeamId")
        is_home_shooter = team_id == home_team_id
        x_coord, y_coord = details.get("xCoord"), details.get("yCoord")
        distance, angle = _shot_geometry(x_coord, y_coord)
        skaters_for, skaters_against, is_empty_net = _strength(
            play.get("situationCode"), is_home_shooter
        )
        period = play.get("periodDescriptor") or {}
        # A blocked shot records the blocker as the event owner, so its
        # shooter and goalie fields are not comparable to other attempts.
        shooter_id = details.get("shootingPlayerId") or details.get("scoringPlayerId")

        rows.append(
            {
                "game_id": game_id,
                "event_id": play.get("eventId"),
                "period": period.get("number"),
                "period_type": period.get("periodType"),
                "time_in_period": play.get("timeInPeriod"),
                "team_id": team_id,
                "is_home_shooter": is_home_shooter,
                "shooter_id": shooter_id,
                "goalie_id": details.get("goalieInNetId"),
                "event_type": event_type,
                "shot_type": details.get("shotType"),
                "zone_code": details.get("zoneCode"),
                "x_coord": x_coord,
                "y_coord": y_coord,
                "distance": distance,
                "angle": angle,
                "skaters_for": skaters_for,
                "skaters_against": skaters_against,
                "is_even_strength": skaters_for == skaters_against
                if skaters_for is not None
                else None,
                "is_empty_net": is_empty_net,
                "is_goal": event_type == "goal",
            }
        )

    return pd.DataFrame(rows, columns=list(SHOT_COLUMNS))


def _already_collected(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path, usecols=["game_id"], dtype={"game_id": str})
    return set(existing["game_id"].unique())


def collect_shots(
    game_ids: Sequence[str] | Iterable[str],
    output_path: str | Path,
    session: requests.Session | None = None,
    checkpoint_every: int = 200,
    request_delay_seconds: float = 0.05,
) -> int:
    """Download play-by-play for each game and append its shot rows.

    The run is resumable: games already present in the output are skipped, so
    an interrupted collection continues instead of starting over.
    """

    from nhl_predictor.nhl_api import build_session, fetch_game_center_feed

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = _already_collected(output)
    client = session or build_session()

    pending = [str(game_id) for game_id in game_ids if str(game_id) not in collected]
    buffer: list[pd.DataFrame] = []
    written = 0
    failed: list[str] = []

    for index, game_id in enumerate(pending, start=1):
        try:
            payload = fetch_game_center_feed(game_id, "play-by-play", client)
        except requests.RequestException as error:
            # Recorded rather than swallowed: a silent gap looks identical to a
            # game that genuinely had no shots.
            failed.append(game_id)
            print(f"failed {game_id}: {error}", flush=True)
            continue
        buffer.append(extract_shots(payload))
        time.sleep(request_delay_seconds)

        if buffer and (index % checkpoint_every == 0 or index == len(pending)):
            chunk = pd.concat(buffer, ignore_index=True)
            chunk.to_csv(output, mode="a", header=not output.exists(), index=False)
            written += len(chunk)
            buffer.clear()
            print(
                f"{index}/{len(pending)} games, {written} shots written, {len(failed)} failed",
                flush=True,
            )

    if failed:
        failure_log = output.with_suffix(".failed.txt")
        failure_log.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} games failed; ids written to {failure_log}", flush=True)
    return written
