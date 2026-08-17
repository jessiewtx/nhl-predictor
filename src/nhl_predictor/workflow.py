"""Daily point-in-time forecast orchestration."""

from __future__ import annotations

import pandas as pd

from nhl_predictor.commitment import commit_predictions
from nhl_predictor.cutoffs import with_forecast_cutoff
from nhl_predictor.features import build_pregame_features, make_training_frame
from nhl_predictor.ledger import ObservationStore
from nhl_predictor.predictor import NhlPredictor

MODEL_VERSION = "logistic-poisson-cutoff-v1"


def generate_forecasts(
    completed_games: pd.DataFrame,
    scheduled_games: pd.DataFrame,
    store: ObservationStore,
    *,
    forecast_kind: str,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Score games and ledger the exact snapshot/model that produced each row."""

    scheduled = with_forecast_cutoff(scheduled_games, forecast_kind)
    history = with_forecast_cutoff(completed_games, forecast_kind)
    all_games = pd.concat([history, scheduled], ignore_index=True)
    all_features = build_pregame_features(all_games)
    training = make_training_frame(history)
    prediction_rows: list[pd.DataFrame] = []

    for cutoff, group in scheduled.groupby("prediction_cutoff_utc", sort=True):
        train_as_of_cutoff = training[training["result_available_at_utc"] <= cutoff]
        model = NhlPredictor().fit(train_as_of_cutoff)
        game_ids = group["game_id"].astype(str)
        features = all_features[all_features["game_id"].astype(str).isin(game_ids)]
        predictions = model.predict_features(features)
        snapshot = store.materialize_snapshot(cutoff)
        recorded = store.record_predictions(
            predictions,
            snapshot_id=snapshot.snapshot_id,
            model_version=model_version,
            forecast_kind=forecast_kind,
            cutoff_utc=cutoff,
        )
        # Committed before the game is played, so the forecast cannot be
        # quietly improved once the result is known.
        commitment = commit_predictions(
            predictions,
            forecast_kind=forecast_kind,
            cutoff_utc=str(cutoff),
            model_version=model_version,
            log_path=store.root / "commitments.jsonl",
        )
        recorded["commitment_hash"] = commitment.entry_hash
        prediction_rows.append(recorded)

    if not prediction_rows:
        return pd.DataFrame()
    return pd.concat(prediction_rows, ignore_index=True)


def material_prediction_changes(
    previous: pd.DataFrame, current: pd.DataFrame, probability_threshold: float = 0.05
) -> pd.DataFrame:
    """Return changes worth notifying on after a final forecast refresh."""

    prior = previous.loc[:, ["game_id", "home_win_probability"]].rename(
        columns={"home_win_probability": "previous_home_win_probability"}
    )
    changed = current.merge(prior, on="game_id", how="left", validate="one_to_one")
    changed["probability_change"] = (
        changed["home_win_probability"] - changed["previous_home_win_probability"]
    )
    changed["winner_changed"] = (
        changed["home_win_probability"] >= 0.5
    ) != (changed["previous_home_win_probability"] >= 0.5)
    return changed[
        changed["winner_changed"] | (changed["probability_change"].abs() >= probability_threshold)
    ].copy()
