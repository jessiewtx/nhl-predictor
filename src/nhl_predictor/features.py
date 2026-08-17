"""Generate pregame features without exposing a game's outcome to itself."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nhl_predictor.schema import FEATURE_COLUMNS, REQUIRED_GAME_COLUMNS

LEAGUE_AVERAGE_GOALS = 3.1
LEAGUE_AVERAGE_EXPECTED_GOALS = 2.8
PRIOR_GAMES = 10
INITIAL_ELO = 1500.0
HOME_ICE_ELO = 35.0
ELO_K = 20.0
OFFSEASON_ELO_REVERSION = 0.5
DEFAULT_RESULT_SETTLEMENT_DELAY = pd.Timedelta(hours=4)
FINAL_FORECAST_LEAD = pd.Timedelta(minutes=30)


@dataclass
class TeamState:
    elo: float = INITIAL_ELO
    games: int = 0
    goals_for: int = 0
    goals_against: int = 0
    expected_goals_for: float = 0.0
    expected_goals_against: float = 0.0
    expected_goal_games: int = 0
    last_game_date: pd.Timestamp | None = None

    @property
    def goal_rate(self) -> float:
        return (self.goals_for + LEAGUE_AVERAGE_GOALS * PRIOR_GAMES) / (self.games + PRIOR_GAMES)

    @property
    def goal_allowed_rate(self) -> float:
        return (self.goals_against + LEAGUE_AVERAGE_GOALS * PRIOR_GAMES) / (
            self.games + PRIOR_GAMES
        )

    @property
    def expected_goal_rate(self) -> float:
        """Chance quality created per game, smoothed toward the league average.

        Without shot data this stays at the prior for every team, so the
        feature is inert rather than quietly mimicking actual goals.
        """

        total = self.expected_goals_for + LEAGUE_AVERAGE_EXPECTED_GOALS * PRIOR_GAMES
        return total / (self.expected_goal_games + PRIOR_GAMES)

    @property
    def expected_goal_allowed_rate(self) -> float:
        total = self.expected_goals_against + LEAGUE_AVERAGE_EXPECTED_GOALS * PRIOR_GAMES
        return total / (self.expected_goal_games + PRIOR_GAMES)


def _rest_days(last_game: pd.Timestamp | None, game_day: pd.Timestamp) -> float:
    if last_game is None:
        return 3.0
    return float(min(max((game_day - last_game).days - 1, 0), 7))


def _validate_games(games: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_GAME_COLUMNS) - set(games.columns)
    if missing:
        raise ValueError(f"Games are missing required columns: {sorted(missing)}")

    normalized = games.copy()
    normalized["start_time_utc"] = pd.to_datetime(normalized["start_time_utc"], utc=True)
    normalized["game_day"] = normalized["start_time_utc"].dt.normalize()
    completed = normalized["home_score"].notna() & normalized["away_score"].notna()
    if "result_available_at_utc" in normalized.columns:
        normalized["result_available_at_utc"] = pd.to_datetime(
            normalized["result_available_at_utc"], utc=True
        )
    else:
        normalized["result_available_at_utc"] = pd.Series(
            pd.NaT, index=normalized.index, dtype="datetime64[ns, UTC]"
        )
    # Historic schedule data has no official wall-clock final time. Four hours
    # after puck drop is deliberately conservative: it can hide valid signal,
    # but cannot expose a score before a later prediction could know it.
    normalized.loc[completed & normalized["result_available_at_utc"].isna(), "result_available_at_utc"] = (
        normalized.loc[completed & normalized["result_available_at_utc"].isna(), "start_time_utc"]
        + DEFAULT_RESULT_SETTLEMENT_DELAY
    )
    if "prediction_cutoff_utc" in normalized.columns:
        normalized["prediction_cutoff_utc"] = pd.to_datetime(
            normalized["prediction_cutoff_utc"], utc=True
        )
    else:
        normalized["prediction_cutoff_utc"] = normalized["start_time_utc"] - FINAL_FORECAST_LEAD
    if "season" in normalized.columns:
        normalized["season_id"] = normalized["season"].astype("string")
    else:
        # NHL seasons begin in the second half of a calendar year. This also
        # identifies the shortened 2020-21 season as season 2020.
        normalized["season_id"] = normalized["game_day"].map(
            lambda game_day: str(game_day.year if game_day.month >= 7 else game_day.year - 1)
        )
    if normalized["game_id"].duplicated().any():
        raise ValueError("game_id must be unique")
    return normalized.sort_values(["prediction_cutoff_utc", "start_time_utc", "game_id"]).reset_index(
        drop=True
    )


def _reset_for_new_season(team_state: dict[str, TeamState]) -> None:
    for state in team_state.values():
        state.elo = INITIAL_ELO + OFFSEASON_ELO_REVERSION * (state.elo - INITIAL_ELO)
        state.games = 0
        state.goals_for = 0
        state.goals_against = 0
        state.expected_goals_for = 0.0
        state.expected_goals_against = 0.0
        state.expected_goal_games = 0
        state.last_game_date = None


def _apply_result(
    game: pd.Series,
    team_state: dict[str, TeamState],
    head_to_head: dict[tuple[str, str], list[int]],
) -> None:
    home = str(game["home_team"])
    away = str(game["away_team"])
    home_score, away_score = int(game["home_score"]), int(game["away_score"])
    if home_score == away_score:
        raise ValueError(f"Completed NHL game {game['game_id']} cannot end tied")
    home_state = team_state.setdefault(home, TeamState())
    away_state = team_state.setdefault(away, TeamState())
    expected_home = 1 / (1 + 10 ** (-(home_state.elo + HOME_ICE_ELO - away_state.elo) / 400))
    home_win = int(home_score > away_score)
    adjustment = ELO_K * (home_win - expected_home)
    home_state.elo += adjustment
    away_state.elo -= adjustment

    game_day = game["start_time_utc"].normalize()
    home_state.games += 1
    home_state.goals_for += home_score
    home_state.goals_against += away_score
    home_state.last_game_date = game_day
    away_state.games += 1
    away_state.goals_for += away_score
    away_state.goals_against += home_score
    away_state.last_game_date = game_day

    home_expected = game.get("home_expected_goals")
    away_expected = game.get("away_expected_goals")
    if pd.notna(home_expected) and pd.notna(away_expected):
        home_state.expected_goals_for += float(home_expected)
        home_state.expected_goals_against += float(away_expected)
        home_state.expected_goal_games += 1
        away_state.expected_goals_for += float(away_expected)
        away_state.expected_goals_against += float(home_expected)
        away_state.expected_goal_games += 1

    for team, opponent, won in ((home, away, home_win), (away, home, 1 - home_win)):
        record = head_to_head.setdefault((team, opponent), [0, 0])
        record[0] += won
        record[1] += 1


def build_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build each feature vector from source results available by its cutoff.

    A completed result enters state only after ``result_available_at_utc``. If
    a historical source lacks that timestamp, a conservative four-hour
    post-start delay is used rather than assuming hindsight availability.
    """

    normalized = _validate_games(games)
    team_state: dict[str, TeamState] = {}
    # Maps (team, opponent) to [wins, games] from the team's perspective.
    head_to_head: dict[tuple[str, str], list[int]] = {}
    output: list[dict[str, object]] = []
    active_season: str | None = None
    completed = normalized.dropna(subset=["home_score", "away_score"]).sort_values(
        ["result_available_at_utc", "game_id"]
    )
    result_index = 0
    completed_rows = [game for _, game in completed.iterrows()]

    for cutoff, cutoff_games in normalized.groupby("prediction_cutoff_utc", sort=True):
        while (
            result_index < len(completed_rows)
            and completed_rows[result_index]["result_available_at_utc"] <= cutoff
        ):
            _apply_result(completed_rows[result_index], team_state, head_to_head)
            result_index += 1

        for _, game in cutoff_games.iterrows():
            season = str(game["season_id"])
            if active_season is not None and season != active_season:
                _reset_for_new_season(team_state)
                head_to_head = {}
            active_season = season

            home = str(game["home_team"])
            away = str(game["away_team"])
            home_state = team_state.setdefault(home, TeamState())
            away_state = team_state.setdefault(away, TeamState())
            h2h_wins, h2h_games = head_to_head.get((home, away), [0, 0])
            game_day = game["start_time_utc"].normalize()

            output.append(
                {
                    "game_id": game["game_id"],
                    "home_elo": home_state.elo,
                    "away_elo": away_state.elo,
                    "elo_difference": home_state.elo + HOME_ICE_ELO - away_state.elo,
                    "home_goal_rate": home_state.goal_rate,
                    "away_goal_rate": away_state.goal_rate,
                    "home_goal_allowed_rate": home_state.goal_allowed_rate,
                    "away_goal_allowed_rate": away_state.goal_allowed_rate,
                    "home_rest_days": _rest_days(home_state.last_game_date, game_day),
                    "away_rest_days": _rest_days(away_state.last_game_date, game_day),
                    "rest_difference": _rest_days(home_state.last_game_date, game_day)
                    - _rest_days(away_state.last_game_date, game_day),
                    "home_h2h_win_rate": (h2h_wins + 1) / (h2h_games + 2),
                    "games_played_difference": home_state.games - away_state.games,
                    "home_expected_goal_rate": home_state.expected_goal_rate,
                    "away_expected_goal_rate": away_state.expected_goal_rate,
                    "home_expected_goal_allowed_rate": home_state.expected_goal_allowed_rate,
                    "away_expected_goal_allowed_rate": away_state.expected_goal_allowed_rate,
                    "expected_goal_rate_difference": (
                        home_state.expected_goal_rate - home_state.expected_goal_allowed_rate
                    )
                    - (away_state.expected_goal_rate - away_state.expected_goal_allowed_rate),
                }
            )

    return pd.DataFrame(output, columns=("game_id", *FEATURE_COLUMNS))


def make_training_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe features and targets for completed games."""

    features = build_pregame_features(games)
    training = _validate_games(games).merge(features, on="game_id", validate="one_to_one")
    completed = training.dropna(subset=["home_score", "away_score"]).copy()
    completed["home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    return completed
