"""Wire contract for pregame status extraction output.

Parsing is deliberately strict. A model that emits an unexpected key or a
status outside the enum has failed, and silently coercing that output would
hide the failure from every downstream forecast.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from nhl_predictor.schema import PlayerStatus, SourceTier

CLAIM_KEYS = frozenset({"player_name", "status", "evidence"})
RESULT_KEYS = frozenset(
    {"team", "as_of_utc", "player_statuses", "confirmed_starting_goalie", "source_tier"}
)
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class ExtractionFormatError(ValueError):
    """Raised when model output does not satisfy the wire contract."""


@dataclass(frozen=True)
class PlayerStatusClaim:
    player_name: str
    status: PlayerStatus
    evidence: str

    def to_dict(self) -> dict[str, str]:
        return {
            "player_name": self.player_name,
            "status": str(self.status),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ExtractionResult:
    team: str
    as_of_utc: str
    player_statuses: tuple[PlayerStatusClaim, ...]
    confirmed_starting_goalie: str | None
    source_tier: SourceTier

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "as_of_utc": self.as_of_utc,
            "player_statuses": [claim.to_dict() for claim in self.player_statuses],
            "confirmed_starting_goalie": self.confirmed_starting_goalie,
            "source_tier": str(self.source_tier),
        }


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionFormatError(f"{field} must be a non-empty string, got {value!r}")
    return value


def _parse_claim(raw: Any, index: int) -> PlayerStatusClaim:
    if not isinstance(raw, dict):
        raise ExtractionFormatError(f"player_statuses[{index}] must be an object")
    unexpected = set(raw) - CLAIM_KEYS
    if unexpected:
        raise ExtractionFormatError(
            f"player_statuses[{index}] has unexpected keys: {sorted(unexpected)}"
        )
    missing = CLAIM_KEYS - set(raw)
    if missing:
        raise ExtractionFormatError(f"player_statuses[{index}] is missing {sorted(missing)}")

    status_value = _require_string(raw["status"], f"player_statuses[{index}].status")
    try:
        status = PlayerStatus(status_value)
    except ValueError as error:
        allowed = sorted(str(member) for member in PlayerStatus)
        raise ExtractionFormatError(
            f"player_statuses[{index}].status {status_value!r} is not one of {allowed}"
        ) from error

    return PlayerStatusClaim(
        player_name=_require_string(raw["player_name"], f"player_statuses[{index}].player_name"),
        status=status,
        evidence=_require_string(raw["evidence"], f"player_statuses[{index}].evidence"),
    )


def parse_extraction(raw: str | dict[str, Any]) -> ExtractionResult:
    """Parse model output into a validated result, or raise ExtractionFormatError."""

    if isinstance(raw, str):
        try:
            payload = json.loads(_FENCE.sub("", raw))
        except json.JSONDecodeError as error:
            raise ExtractionFormatError(f"output is not valid JSON: {error}") from error
    else:
        payload = raw

    if not isinstance(payload, dict):
        raise ExtractionFormatError("output must be a single JSON object")
    unexpected = set(payload) - RESULT_KEYS
    if unexpected:
        raise ExtractionFormatError(f"output has unexpected keys: {sorted(unexpected)}")
    missing = RESULT_KEYS - set(payload)
    if missing:
        raise ExtractionFormatError(f"output is missing {sorted(missing)}")

    statuses = payload["player_statuses"]
    if not isinstance(statuses, list):
        raise ExtractionFormatError("player_statuses must be a list")

    tier_value = _require_string(payload["source_tier"], "source_tier")
    try:
        tier = SourceTier(tier_value)
    except ValueError as error:
        allowed = sorted(str(member) for member in SourceTier)
        raise ExtractionFormatError(
            f"source_tier {tier_value!r} is not one of {allowed}"
        ) from error

    starter = payload["confirmed_starting_goalie"]
    if starter is not None:
        starter = _require_string(starter, "confirmed_starting_goalie")

    return ExtractionResult(
        team=_require_string(payload["team"], "team"),
        as_of_utc=_require_string(payload["as_of_utc"], "as_of_utc"),
        player_statuses=tuple(_parse_claim(claim, i) for i, claim in enumerate(statuses)),
        confirmed_starting_goalie=starter,
        source_tier=tier,
    )
