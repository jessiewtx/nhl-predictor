"""Tamper-evident commitments for predictions made before games are played.

A backtest can be audited after the fact, but a live forecast is only worth
something if it cannot be quietly revised once the result is known. Before puck
drop, the predictions are hashed and that digest is appended to a chained log:
every entry carries the previous entry's hash, so altering an old prediction
changes every digest after it.

This does not prove a prediction was good. It proves it was not edited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

GENESIS_HASH = "0" * 64
COMMITTED_FIELDS = ("game_id", "home_win_probability", "expected_home_goals", "expected_away_goals")


@dataclass(frozen=True)
class Commitment:
    committed_at_utc: str
    forecast_kind: str
    cutoff_utc: str
    model_version: str
    game_count: int
    predictions_digest: str
    previous_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def digest_predictions(predictions: pd.DataFrame) -> str:
    """Hash only the fields that constitute the forecast itself.

    Presentation columns are excluded so that reformatting output does not
    invalidate a commitment, while any change to a probability does.
    """

    missing = set(COMMITTED_FIELDS) - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing committed fields: {sorted(missing)}")
    frame = predictions.loc[:, list(COMMITTED_FIELDS)].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.sort_values("game_id").reset_index(drop=True)
    rows = [
        {
            "game_id": row["game_id"],
            "home_win_probability": round(float(row["home_win_probability"]), 6),
            "expected_home_goals": round(float(row["expected_home_goals"]), 4),
            "expected_away_goals": round(float(row["expected_away_goals"]), 4),
        }
        for _, row in frame.iterrows()
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def _last_hash(log_path: Path) -> str:
    if not log_path.exists():
        return GENESIS_HASH
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    if not lines:
        return GENESIS_HASH
    return json.loads(lines[-1])["entry_hash"]


def commit_predictions(
    predictions: pd.DataFrame,
    *,
    forecast_kind: str,
    cutoff_utc: str,
    model_version: str,
    log_path: str | Path = "data/ledger/commitments.jsonl",
) -> Commitment:
    """Append a chained commitment for a set of pregame forecasts."""

    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    previous_hash = _last_hash(log)
    predictions_digest = digest_predictions(predictions)
    committed_at = datetime.now(UTC).isoformat()

    payload = {
        "committed_at_utc": committed_at,
        "forecast_kind": forecast_kind,
        "cutoff_utc": str(cutoff_utc),
        "model_version": model_version,
        "game_count": len(predictions),
        "predictions_digest": predictions_digest,
        "previous_hash": previous_hash,
    }
    entry_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    commitment = Commitment(**payload, entry_hash=entry_hash)

    with log.open("a") as stream:
        stream.write(json.dumps(commitment.to_dict(), sort_keys=True) + "\n")
    return commitment


def verify_chain(log_path: str | Path = "data/ledger/commitments.jsonl") -> tuple[bool, str]:
    """Recompute every hash and confirm the chain is unbroken."""

    log = Path(log_path)
    if not log.exists():
        return True, "no commitments recorded yet"

    previous_hash = GENESIS_HASH
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    for index, entry in enumerate(entries):
        if entry["previous_hash"] != previous_hash:
            return False, f"entry {index} does not follow the previous entry"
        payload = {key: entry[key] for key in entry if key != "entry_hash"}
        expected = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if expected != entry["entry_hash"]:
            return False, f"entry {index} was modified after it was written"
        previous_hash = entry["entry_hash"]
    return True, f"{len(entries)} commitments verified, chain intact"
