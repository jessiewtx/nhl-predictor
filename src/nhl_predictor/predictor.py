"""Models for NHL win probabilities and expected scores."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nhl_predictor.features import build_pregame_features, make_training_frame
from nhl_predictor.schema import FEATURE_COLUMNS


class NhlPredictor:
    """A transparent baseline trained only on pregame feature snapshots."""

    def __init__(self, feature_columns: Sequence[str] | None = None) -> None:
        self.feature_columns = tuple(feature_columns) if feature_columns else FEATURE_COLUMNS
        if not self.feature_columns:
            raise ValueError("At least one feature column is required.")
        unknown = set(self.feature_columns) - set(FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown feature columns: {sorted(unknown)}")
        self.win_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=1_000)),
            ]
        )
        self.home_goals_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", PoissonRegressor(alpha=1.0, max_iter=1_000)),
            ]
        )
        self.away_goals_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", PoissonRegressor(alpha=1.0, max_iter=1_000)),
            ]
        )
        self.is_fitted = False

    def fit(self, training_frame: pd.DataFrame) -> NhlPredictor:
        """Fit models from the output of ``make_training_frame``."""

        if len(training_frame) < 30:
            raise ValueError("At least 30 completed games are required to fit a model.")
        x = training_frame.loc[:, list(self.feature_columns)]
        self.win_model.fit(x, training_frame["home_win"])
        self.home_goals_model.fit(x, training_frame["home_score"])
        self.away_goals_model.fit(x, training_frame["away_score"])
        self.is_fitted = True
        return self

    def fit_from_games(self, completed_games: pd.DataFrame) -> NhlPredictor:
        return self.fit(make_training_frame(completed_games))

    def predict_features(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit before requesting predictions.")
        x = feature_frame.loc[:, list(self.feature_columns)]
        result = feature_frame.loc[:, ["game_id"]].copy()
        result["home_win_probability"] = self.win_model.predict_proba(x)[:, 1]
        result["expected_home_goals"] = self.home_goals_model.predict(x)
        result["expected_away_goals"] = self.away_goals_model.predict(x)
        return result

    def predict_games(
        self, completed_games: pd.DataFrame, scheduled_games: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict scheduled games using the state produced by completed games.

        Scheduled rows must use the canonical game schema and leave both score
        columns empty. The feature builder preserves that distinction.
        """

        self.fit_from_games(completed_games)
        all_games = pd.concat([completed_games, scheduled_games], ignore_index=True)
        all_features = build_pregame_features(all_games)
        scheduled_features = all_features[all_features["game_id"].isin(scheduled_games["game_id"])]
        return self.predict_features(scheduled_features)
