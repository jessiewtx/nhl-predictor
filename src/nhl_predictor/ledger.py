"""Append-only observation, snapshot, and prediction ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nhl_predictor.schema import Observation, SourceTier


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return utc_now()
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("All timestamps must include an explicit timezone.")
    return timestamp.tz_convert("UTC")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    cutoff_utc: str
    observation_ids: tuple[str, ...]
    manifest_path: Path


class ObservationStore:
    """Local append-only store; suitable for a mounted Modal volume later."""

    def __init__(self, root: str | Path = "data/ledger") -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.snapshots_dir = self.root / "snapshots"
        self.observations_path = self.root / "observations.jsonl"
        self.predictions_path = self.root / "predictions.jsonl"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        payload: object,
        *,
        source_kind: str,
        source_url: str,
        source_tier: SourceTier = SourceTier.OFFICIAL,
        fetched_at_utc: str | pd.Timestamp | None = None,
        published_at_utc: str | pd.Timestamp | None = None,
        available_at_utc: str | pd.Timestamp | None = None,
        game_id: str | None = None,
        team: str | None = None,
        player_id: str | None = None,
    ) -> Observation:
        """Persist a raw source payload and one immutable availability record."""

        fetched_at = _timestamp(fetched_at_utc)
        published_at = _timestamp(published_at_utc) if published_at_utc is not None else None
        available_at = _timestamp(available_at_utc) if available_at_utc is not None else fetched_at
        if available_at > fetched_at:
            raise ValueError("available_at_utc cannot be after fetched_at_utc.")

        canonical_payload = _canonical_json(payload)
        payload_hash = sha256(canonical_payload.encode()).hexdigest()
        raw_path = self.raw_dir / f"{payload_hash}.json"
        if not raw_path.exists():
            raw_path.write_text(canonical_payload + "\n")

        observation_id = sha256(
            f"{source_url}|{available_at.isoformat()}|{payload_hash}|{uuid4()}".encode()
        ).hexdigest()[:24]
        observation = Observation(
            observation_id=observation_id,
            source_kind=source_kind,
            source_tier=source_tier,
            source_url=source_url,
            fetched_at_utc=fetched_at.isoformat(),
            published_at_utc=published_at.isoformat() if published_at is not None else None,
            available_at_utc=available_at.isoformat(),
            payload_hash=payload_hash,
            game_id=str(game_id) if game_id is not None else None,
            team=team,
            player_id=str(player_id) if player_id is not None else None,
        )
        with self.observations_path.open("a") as stream:
            stream.write(_canonical_json(observation.to_dict()) + "\n")
        return observation

    def observations_as_of(self, cutoff_utc: str | pd.Timestamp) -> pd.DataFrame:
        """Return only records the system could have known by the cutoff."""

        if not self.observations_path.exists():
            return pd.DataFrame()
        cutoff = _timestamp(cutoff_utc)
        rows = [json.loads(line) for line in self.observations_path.read_text().splitlines() if line]
        observations = pd.DataFrame(rows)
        available = pd.to_datetime(observations["available_at_utc"], utc=True)
        return observations.loc[available <= cutoff].copy()

    def materialize_snapshot(self, cutoff_utc: str | pd.Timestamp) -> Snapshot:
        """Create a deterministic manifest for all observations usable at a cutoff."""

        cutoff = _timestamp(cutoff_utc)
        observations = self.observations_as_of(cutoff)
        observation_ids = tuple(sorted(observations.get("observation_id", pd.Series(dtype=str)).tolist()))
        digest_input = _canonical_json(
            {"cutoff_utc": cutoff.isoformat(), "observation_ids": observation_ids}
        )
        snapshot_id = sha256(digest_input.encode()).hexdigest()[:24]
        manifest_path = self.snapshots_dir / f"{snapshot_id}.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {
                        "snapshot_id": snapshot_id,
                        "cutoff_utc": cutoff.isoformat(),
                        "observation_ids": observation_ids,
                        "created_at_utc": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                )
                + "\n"
            )
        return Snapshot(snapshot_id, cutoff.isoformat(), observation_ids, manifest_path)

    def record_predictions(
        self,
        predictions: pd.DataFrame,
        *,
        snapshot_id: str,
        model_version: str,
        forecast_kind: str,
        cutoff_utc: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Append predictions with the precise data/model provenance that made them."""

        cutoff = _timestamp(cutoff_utc).isoformat()
        recorded = predictions.copy()
        recorded["prediction_id"] = [uuid4().hex for _ in range(len(recorded))]
        recorded["snapshot_id"] = snapshot_id
        recorded["model_version"] = model_version
        recorded["forecast_kind"] = forecast_kind
        recorded["cutoff_utc"] = cutoff
        with self.predictions_path.open("a") as stream:
            for row in recorded.to_dict(orient="records"):
                stream.write(_canonical_json(row) + "\n")
        return recorded
