"""Adversarial checks that try to prove the pipeline is cheating.

Good backtest numbers are easy to produce by accident. The usual cause is
leakage: some path by which a game's own result, or a later result, reaches its
features. These checks are designed to fail loudly if that happens, so the
reported metrics can be believed rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhl_predictor.backtest import summarize_backtest, walk_forward_backtest
from nhl_predictor.features import build_pregame_features
from nhl_predictor.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def permute_outcomes(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Randomly reassign which game got which final score.

    Team strength no longer explains anything, so a leak-free pipeline must
    lose all skill. Score pairs move together, which preserves the league's
    scoring distribution and home-win rate; only the link between the teams
    and the result is destroyed.
    """

    rng = np.random.default_rng(seed)
    shuffled = games.copy()
    order = rng.permutation(len(shuffled))
    scores = shuffled.loc[:, ["home_score", "away_score"]].to_numpy()[order]
    shuffled["home_score"] = scores[:, 0]
    shuffled["away_score"] = scores[:, 1]
    return shuffled


def permutation_check(
    games: pd.DataFrame, minimum_training_games: int = 500, tolerance: float = 0.02
) -> CheckResult:
    """Skill on shuffled outcomes is evidence of leakage, not intelligence."""

    predictions = walk_forward_backtest(
        permute_outcomes(games), minimum_training_games=minimum_training_games
    )
    summary = summarize_backtest(predictions)
    lift = float(summary["accuracy_lift_over_always_home"])
    log_loss = float(summary["log_loss"])
    passed = lift <= tolerance and log_loss >= 0.68
    return CheckResult(
        "permuted_outcomes_show_no_skill",
        passed,
        f"accuracy {summary['accuracy']:.4f} vs always-home "
        f"{summary['always_home_accuracy']:.4f} (lift {lift:+.4f}), log loss {log_loss:.4f}"
        + ("" if passed else "  <-- skill on random labels means the pipeline leaks"),
    )


def prefix_invariance_check(games: pd.DataFrame, prefix_fraction: float = 0.5) -> CheckResult:
    """Deleting future games must not change any past game's features.

    This is the definitive point-in-time test. If a feature for an October game
    shifts when March games are removed, then March was informing October.
    """

    ordered = games.sort_values("start_time_utc").reset_index(drop=True)
    cutoff = int(len(ordered) * prefix_fraction)
    prefix = ordered.iloc[:cutoff]

    full_features = build_pregame_features(ordered).set_index("game_id")
    prefix_features = build_pregame_features(prefix).set_index("game_id")

    shared = prefix_features.index.intersection(full_features.index)
    left = full_features.loc[shared, list(FEATURE_COLUMNS)].astype(float)
    right = prefix_features.loc[shared, list(FEATURE_COLUMNS)].astype(float)
    difference = (left - right).abs().to_numpy()
    worst = float(np.nanmax(difference)) if difference.size else 0.0
    passed = worst < 1e-9
    return CheckResult(
        "past_features_ignore_future_games",
        passed,
        f"compared {len(shared)} games on {len(FEATURE_COLUMNS)} features; "
        f"largest difference {worst:.2e}"
        + ("" if passed else "  <-- future games changed past features"),
    )


def determinism_check(games: pd.DataFrame) -> CheckResult:
    """The same inputs must produce byte-identical features."""

    first = build_pregame_features(games)
    second = build_pregame_features(games)
    passed = first.equals(second)
    return CheckResult(
        "features_are_reproducible",
        passed,
        "two runs produced identical features"
        if passed
        else "two runs disagreed, so results are not reproducible",
    )


def label_reconstruction_check(games: pd.DataFrame) -> CheckResult:
    """No single feature should correlate near-perfectly with the outcome.

    A feature that all but announces the winner is the signature of a target
    accidentally leaking into the inputs.
    """

    from nhl_predictor.features import make_training_frame

    frame = make_training_frame(games)
    correlations = {
        column: abs(float(np.corrcoef(frame[column], frame["home_win"])[0, 1]))
        for column in FEATURE_COLUMNS
        if frame[column].std() > 0
    }
    worst_feature = max(correlations, key=correlations.get)
    worst_value = correlations[worst_feature]
    passed = worst_value < 0.5
    return CheckResult(
        "no_feature_reveals_the_outcome",
        passed,
        f"strongest single-feature correlation is {worst_feature} at {worst_value:.3f}"
        + ("" if passed else "  <-- a feature is close to announcing the winner"),
    )


def run_all_checks(games: pd.DataFrame, minimum_training_games: int = 500) -> list[CheckResult]:
    return [
        prefix_invariance_check(games),
        determinism_check(games),
        label_reconstruction_check(games),
        permutation_check(games, minimum_training_games),
    ]
