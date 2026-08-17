"""An expected-goals model trained on our own archived play-by-play.

Expected goals answer "how good were the chances this team generated", which is
far steadier than the goals actually scored. That stability is the point: goal
totals swing on deflections, while chance quality persists and predicts.

Leakage matters twice over here. The xG model is itself fitted on historical
shots, so it is trained once on the earliest seasons and then frozen. If it were
refit on all data, every backtest would score chances using a model that had
already seen the outcomes it was being asked to anticipate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Blocked attempts are recorded at the blocker's location rather than the
# shooter's, and carry no shot type, so they are excluded rather than modeled
# with coordinates that describe a different event.
UNBLOCKED_EVENTS = frozenset({"goal", "shot-on-goal", "missed-shot"})

# Marks games the expected-goals model was fitted on. Backtests train on them
# but must not report metrics from them.
EVALUATION_EXCLUDED_COLUMN = "in_xg_training_window"

NUMERIC_FEATURES = ["distance", "angle", "skater_differential"]
CATEGORICAL_FEATURES = ["shot_type", "event_strength"]


@dataclass(frozen=True)
class ExpectedGoalsReport:
    training_shots: int
    evaluation_shots: int
    log_loss: float
    roc_auc: float
    base_rate: float


def prepare_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Filter to modelable attempts and derive the model's input columns."""

    frame = shots[shots["event_type"].isin(UNBLOCKED_EVENTS)].copy()
    frame = frame.dropna(subset=["distance", "angle"])
    frame["shot_type"] = frame["shot_type"].fillna("unknown").astype(str)
    skaters_for = frame["skaters_for"].fillna(5)
    skaters_against = frame["skaters_against"].fillna(5)
    frame["skater_differential"] = skaters_for - skaters_against
    frame["event_strength"] = np.where(
        frame["is_empty_net"].fillna(False),
        "empty_net",
        np.where(
            frame["skater_differential"] > 0,
            "power_play",
            np.where(frame["skater_differential"] < 0, "shorthanded", "even"),
        ),
    )
    frame["is_goal"] = frame["is_goal"].astype(int)
    return frame


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            CATEGORICAL_FEATURES,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4, learning_rate=0.08, max_iter=250, random_state=0
                ),
            ),
        ]
    )


class ExpectedGoalsModel:
    """Fits once on early seasons, then scores every later shot unchanged."""

    def __init__(self) -> None:
        self.pipeline = build_model()
        self.is_fitted = False

    def fit(self, training_shots: pd.DataFrame) -> ExpectedGoalsModel:
        frame = prepare_shots(training_shots)
        if len(frame) < 1_000:
            raise ValueError("At least 1,000 unblocked attempts are required to fit xG.")
        features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        self.pipeline.fit(features, frame["is_goal"])
        self.is_fitted = True
        return self

    def predict(self, shots: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Fit the expected-goals model before scoring shots.")
        frame = prepare_shots(shots)
        features = frame[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
        return pd.Series(
            self.pipeline.predict_proba(features)[:, 1], index=frame.index, name="expected_goal"
        )

    def evaluate(self, holdout_shots: pd.DataFrame) -> ExpectedGoalsReport:
        frame = prepare_shots(holdout_shots)
        predictions = self.predict(holdout_shots)
        actual = frame["is_goal"]
        return ExpectedGoalsReport(
            training_shots=0,
            evaluation_shots=len(frame),
            log_loss=float(log_loss(actual, predictions.clip(1e-6, 1 - 1e-6))),
            roc_auc=float(roc_auc_score(actual, predictions)),
            base_rate=float(actual.mean()),
        )


def team_game_expected_goals(shots: pd.DataFrame, expected_goals: pd.Series) -> pd.DataFrame:
    """Aggregate shot-level xG into one row per game per side."""

    frame = prepare_shots(shots)
    frame = frame.assign(expected_goal=expected_goals)
    grouped = (
        frame.groupby(["game_id", "is_home_shooter"], sort=True)
        .agg(
            expected_goals_for=("expected_goal", "sum"),
            attempts=("expected_goal", "size"),
            goals=("is_goal", "sum"),
        )
        .reset_index()
    )
    return grouped


def season_of(game_id: str) -> str:
    """NHL game ids begin with the season's starting year."""

    return str(game_id)[:4]


def build_expected_goals_dataset(
    shots: pd.DataFrame, games: pd.DataFrame, training_seasons: Sequence[str]
) -> tuple[pd.DataFrame, ExpectedGoalsReport]:
    """Fit xG on early seasons only, then score every game with that frozen model.

    Holding the model fixed is what keeps later backtests honest: chances from
    2024 are graded by a model that never saw how they turned out.
    """

    shots = shots.copy()
    shots["game_id"] = shots["game_id"].astype(str)
    shots["season"] = shots["game_id"].map(season_of)
    training_seasons = tuple(str(season) for season in training_seasons)

    training = shots[shots["season"].isin(training_seasons)]
    holdout = shots[~shots["season"].isin(training_seasons)]
    if training.empty:
        raise ValueError(f"No shots found for training seasons {training_seasons}")

    model = ExpectedGoalsModel().fit(training)
    report = model.evaluate(holdout if not holdout.empty else training)
    report = ExpectedGoalsReport(
        training_shots=len(prepare_shots(training)),
        evaluation_shots=report.evaluation_shots,
        log_loss=report.log_loss,
        roc_auc=report.roc_auc,
        base_rate=report.base_rate,
    )

    totals = team_game_expected_goals(shots, model.predict(shots))
    enriched = attach_team_expected_goals(games, totals)
    # The xG model saw these seasons' outcomes, so their xG features are not a
    # pregame quantity. They remain usable as history for later predictions,
    # because this model existed before those later seasons began, but scoring
    # the model on them would be grading it on its own training data.
    enriched[EVALUATION_EXCLUDED_COLUMN] = (
        enriched["game_id"].astype(str).map(season_of).isin(training_seasons)
    )
    return enriched, report


def attach_team_expected_goals(games: pd.DataFrame, team_totals: pd.DataFrame) -> pd.DataFrame:
    """Add per-game home and away expected goals to the canonical games frame.

    Games without shot data keep null xG, which leaves those features resting
    at the league-average prior instead of inventing a value.
    """

    wide = team_totals.pivot(
        index="game_id", columns="is_home_shooter", values="expected_goals_for"
    )
    wide = wide.rename(columns={True: "home_expected_goals", False: "away_expected_goals"})
    for column in ("home_expected_goals", "away_expected_goals"):
        if column not in wide.columns:
            wide[column] = np.nan
    wide = wide.loc[:, ["home_expected_goals", "away_expected_goals"]].reset_index()
    wide["game_id"] = wide["game_id"].astype(str)

    merged = games.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    return merged.merge(wide, on="game_id", how="left", validate="one_to_one")
