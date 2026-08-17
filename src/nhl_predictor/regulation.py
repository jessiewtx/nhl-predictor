"""Model regulation and overtime separately instead of one binary outcome.

Roughly a fifth of NHL games are tied after sixty minutes and settled in
three-on-three overtime or a shootout, where the winner is close to a coin flip
regardless of team quality. Training a single binary classifier on those games
asks it to explain noise, which costs calibration on the games that are
genuinely predictable.

This model instead predicts the regulation result as three outcomes, then
resolves ties with a separate, deliberately weak overtime model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nhl_predictor.features import make_training_frame
from nhl_predictor.schema import FEATURE_COLUMNS

REGULATION_OUTCOMES = ("away", "tie", "home")
BEYOND_REGULATION = frozenset({"OT", "SO"})


def label_regulation_outcome(games: pd.DataFrame) -> pd.Series:
    """Label each completed game by how it stood after sixty minutes.

    Any game that reached overtime was tied in regulation, whatever the final
    score shows.
    """

    if "last_period_type" not in games.columns:
        raise ValueError(
            "last_period_type is required; re-collect schedules to record OT and SO games."
        )
    beyond = games["last_period_type"].isin(BEYOND_REGULATION)
    home_won = games["home_score"] > games["away_score"]
    return pd.Series(
        np.where(beyond, "tie", np.where(home_won, "home", "away")),
        index=games.index,
        name="regulation_outcome",
    )


class RegulationPredictor:
    """Three-way regulation model plus a separate overtime tiebreak."""

    def __init__(self, feature_columns: Sequence[str] | None = None) -> None:
        self.feature_columns = tuple(feature_columns) if feature_columns else FEATURE_COLUMNS
        self.regulation_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=1_000)),
            ]
        )
        # Overtime is nearly a coin flip, so it gets one heavily regularized
        # feature rather than the full set it would happily overfit.
        self.overtime_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.05, max_iter=1_000)),
            ]
        )
        self.home_goals_model = Pipeline(
            [("scale", StandardScaler()), ("model", PoissonRegressor(alpha=1.0, max_iter=1_000))]
        )
        self.away_goals_model = Pipeline(
            [("scale", StandardScaler()), ("model", PoissonRegressor(alpha=1.0, max_iter=1_000))]
        )
        self.overtime_home_rate = 0.5
        self.is_fitted = False

    def fit(self, training_frame: pd.DataFrame) -> RegulationPredictor:
        if len(training_frame) < 30:
            raise ValueError("At least 30 completed games are required to fit a model.")
        frame = training_frame.copy()
        frame["regulation_outcome"] = label_regulation_outcome(frame)
        features = frame.loc[:, list(self.feature_columns)]

        self.regulation_model.fit(features, frame["regulation_outcome"])
        self.home_goals_model.fit(features, frame["home_score"])
        self.away_goals_model.fit(features, frame["away_score"])

        overtime = frame[frame["regulation_outcome"] == "tie"]
        self.overtime_home_rate = (
            float((overtime["home_score"] > overtime["away_score"]).mean())
            if len(overtime)
            else 0.5
        )
        if len(overtime) >= 100:
            self.overtime_model.fit(
                overtime.loc[:, ["elo_difference"]],
                (overtime["home_score"] > overtime["away_score"]).astype(int),
            )
        else:
            self.overtime_model = None
        self.is_fitted = True
        return self

    def fit_from_games(self, completed_games: pd.DataFrame) -> RegulationPredictor:
        return self.fit(make_training_frame(completed_games))

    def predict_features(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit before requesting predictions.")
        features = feature_frame.loc[:, list(self.feature_columns)]
        regulation = self.regulation_model.predict_proba(features)
        classes = list(self.regulation_model.named_steps["model"].classes_)
        probability = {
            outcome: regulation[:, classes.index(outcome)]
            for outcome in classes
        }
        home_regulation = probability.get("home", np.zeros(len(features)))
        tie = probability.get("tie", np.zeros(len(features)))

        if self.overtime_model is not None:
            overtime_home = self.overtime_model.predict_proba(
                feature_frame.loc[:, ["elo_difference"]]
            )[:, 1]
        else:
            overtime_home = np.full(len(features), self.overtime_home_rate)

        result = feature_frame.loc[:, ["game_id"]].copy()
        result["home_win_probability"] = home_regulation + tie * overtime_home
        result["regulation_home_win_probability"] = home_regulation
        result["overtime_probability"] = tie
        result["expected_home_goals"] = self.home_goals_model.predict(features)
        result["expected_away_goals"] = self.away_goals_model.predict(features)
        return result
