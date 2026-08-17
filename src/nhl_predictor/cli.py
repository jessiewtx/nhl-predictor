"""Command-line entry points for collection and leakage-safe backtests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from nhl_predictor.backtest import run_ablation, summarize_backtest, walk_forward_backtest
from nhl_predictor.espn_api import collect_league_injuries
from nhl_predictor.expected_goals import build_expected_goals_dataset
from nhl_predictor.ledger import ObservationStore
from nhl_predictor.nhl_api import (
    capture_official_observations,
    collect_schedule,
    fetch_schedule_day,
)
from nhl_predictor.predictor import NhlPredictor
from nhl_predictor.publish import build_snapshot, write_snapshot
from nhl_predictor.shots import collect_shots
from nhl_predictor.verification import run_all_checks
from nhl_predictor.workflow import generate_forecasts, material_prediction_changes


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _collect(args: argparse.Namespace) -> None:
    games = collect_schedule(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output, index=False)
    print(f"Saved {len(games)} completed regular-season games to {output}")


def _combine(args: argparse.Namespace) -> None:
    games = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    if games["game_id"].duplicated().any():
        raise ValueError("Input files contain duplicate game_id values.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    games.sort_values("start_time_utc").to_csv(output, index=False)
    print(f"Saved {len(games)} combined games to {output}")


def _backtest(args: argparse.Namespace) -> None:
    games = pd.read_csv(args.games)
    predictions = walk_forward_backtest(games, args.minimum_training_games, args.forecast_kind)
    summary = summarize_backtest(predictions)
    if args.predictions:
        output = Path(args.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output, index=False)
    print(json.dumps(summary, indent=2))


def _schedule(args: argparse.Namespace) -> None:
    games = fetch_schedule_day(args.date)
    schedule = games[games["game_date"] == args.date.isoformat()].copy()
    # A schedule is a pregame artifact even if this command is run retrospectively.
    schedule["home_score"] = pd.NA
    schedule["away_score"] = pd.NA
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False)
    print(f"Saved {len(schedule)} scheduled regular-season games to {output}")


def _predict(args: argparse.Namespace) -> None:
    completed_games = pd.read_csv(args.history)
    scheduled_games = pd.read_csv(args.schedule)
    predictions = NhlPredictor().predict_games(completed_games, scheduled_games)
    identity = scheduled_games.loc[
        :, ["game_id", "start_time_utc", "away_team", "home_team"]
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity.merge(predictions, on="game_id", validate="one_to_one").to_csv(output, index=False)
    print(f"Saved {len(predictions)} predictions to {output}")


def _capture_official(args: argparse.Namespace) -> None:
    store = ObservationStore(args.ledger)
    games = capture_official_observations(
        args.date, store, include_injury_reports=not args.skip_injury_reports
    )
    print(f"Captured official observations for {len(games)} scheduled regular-season games.")


def _collect_shots(args: argparse.Namespace) -> None:
    games = pd.read_csv(args.games, dtype={"game_id": str})
    written = collect_shots(games["game_id"].tolist(), args.output)
    print(f"Wrote {written} new shot rows to {args.output}")


def _build_xg(args: argparse.Namespace) -> None:
    shots = pd.read_csv(args.shots, dtype={"game_id": str})
    games = pd.read_csv(args.games, dtype={"game_id": str})
    seasons = [season.strip() for season in args.train_seasons.split(",")]
    enriched, report = build_expected_goals_dataset(shots, games, seasons)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    covered = enriched["home_expected_goals"].notna().sum()
    print(json.dumps(asdict(report), indent=2))
    print(f"\n{covered}/{len(enriched)} games have expected goals; wrote {output}")


def _ablate(args: argparse.Namespace) -> None:
    games = pd.read_csv(args.games)
    ladder = [group.strip() for group in args.ladder.split(",")] if args.ladder else None
    table = run_ablation(
        games,
        **({"ladder": ladder} if ladder else {}),
        minimum_training_games=args.minimum_training_games,
        forecast_kind=args.forecast_kind,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved ablation table to {output}")


def _verify(args: argparse.Namespace) -> None:
    games = pd.read_csv(args.games, dtype={"game_id": str})
    results = run_all_checks(games, args.minimum_training_games)
    for result in results:
        print(result)
    failed = [result for result in results if not result.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        raise SystemExit(1)


def _publish(args: argparse.Namespace) -> None:
    backtest = pd.read_csv(args.backtest, dtype={"game_id": str})
    games = pd.read_csv(args.games, dtype={"game_id": str})
    ablation = pd.read_csv(args.ablation) if Path(args.ablation).exists() else None
    upcoming = (
        pd.read_csv(args.upcoming, dtype={"game_id": str})
        if args.upcoming and Path(args.upcoming).exists()
        else None
    )
    snapshot = build_snapshot(backtest, games, ablation, upcoming)
    output = write_snapshot(snapshot, args.output)
    headline = snapshot["headline"]
    print(
        f"Published {headline['games_evaluated']} graded games "
        f"(accuracy {headline['accuracy']:.3f}, log loss {headline['log_loss']:.4f}) to {output}"
    )


def _capture_statuses(args: argparse.Namespace) -> None:
    store = ObservationStore(args.ledger)
    statuses = collect_league_injuries(store)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    statuses.to_csv(output, index=False)
    unavailable = (statuses["status"] == "confirmed_out").sum() if len(statuses) else 0
    print(f"Captured {len(statuses)} player statuses ({unavailable} unavailable) to {output}")


def _forecast(args: argparse.Namespace) -> None:
    completed_games = pd.read_csv(args.history)
    scheduled_games = pd.read_csv(args.schedule)
    predictions = generate_forecasts(
        completed_games,
        scheduled_games,
        ObservationStore(args.ledger),
        forecast_kind=args.forecast_kind,
    )
    identity = scheduled_games.loc[:, ["game_id", "start_time_utc", "away_team", "home_team"]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast = identity.merge(predictions, on="game_id", validate="one_to_one")
    forecast.to_csv(output, index=False)
    if args.previous:
        previous = pd.read_csv(args.previous)
        changes = material_prediction_changes(previous, forecast, args.change_threshold)
        if args.changes_output:
            changes_path = Path(args.changes_output)
            changes_path.parent.mkdir(parents=True, exist_ok=True)
            changes.to_csv(changes_path, index=False)
        print(f"Saved {len(forecast)} forecasts; {len(changes)} changes meet the notification threshold.")
    else:
        print(f"Saved {len(forecast)} {args.forecast_kind} forecasts to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NHL predictor data and evaluation commands")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect completed regular-season games")
    collect.add_argument("--start", type=_parse_date, required=True)
    collect.add_argument("--end", type=_parse_date, required=True)
    collect.add_argument("--output", default="data/raw/nhl_games.csv")
    collect.set_defaults(func=_collect)

    combine = commands.add_parser("combine", help="combine season CSVs into one history file")
    combine.add_argument("inputs", nargs="+", help="completed-game CSV files in any order")
    combine.add_argument("--output", default="data/raw/history.csv")
    combine.set_defaults(func=_combine)

    schedule = commands.add_parser("schedule", help="save a score-free regular-season schedule")
    schedule.add_argument("--date", type=_parse_date, required=True)
    schedule.add_argument("--output", default="data/raw/todays_schedule.csv")
    schedule.set_defaults(func=_schedule)

    backtest = commands.add_parser("backtest", help="run a walk-forward historical backtest")
    backtest.add_argument("--games", default="data/raw/nhl_games.csv")
    backtest.add_argument("--minimum-training-games", type=int, default=300)
    backtest.add_argument("--forecast-kind", choices=("morning", "final"), default="final")
    backtest.add_argument("--predictions", default="data/processed/backtest_predictions.csv")
    backtest.set_defaults(func=_backtest)

    predict = commands.add_parser("predict", help="score scheduled games from completed history")
    predict.add_argument("--history", required=True, help="CSV containing completed games")
    predict.add_argument("--schedule", required=True, help="CSV containing future games with blank scores")
    predict.add_argument("--output", default="data/processed/predictions.csv")
    predict.set_defaults(func=_predict)

    capture = commands.add_parser(
        "capture-official", help="capture immutable official schedule, game, and injury observations"
    )
    capture.add_argument("--date", type=_parse_date, required=True)
    capture.add_argument("--ledger", default="data/ledger")
    capture.add_argument("--skip-injury-reports", action="store_true")
    capture.set_defaults(func=_capture_official)

    shots = commands.add_parser(
        "collect-shots", help="download play-by-play shot events for expected goals"
    )
    shots.add_argument("--games", required=True, help="CSV of games to collect")
    shots.add_argument("--output", default="data/raw/shots.csv")
    shots.set_defaults(func=_collect_shots)

    build_xg = commands.add_parser(
        "build-xg", help="fit expected goals on early seasons and attach it to games"
    )
    build_xg.add_argument("--shots", default="data/raw/shots_2020_2025.csv")
    build_xg.add_argument("--games", default="data/raw/history_2020_2025.csv")
    build_xg.add_argument("--train-seasons", default="2020,2021")
    build_xg.add_argument("--output", default="data/raw/games_with_xg.csv")
    build_xg.set_defaults(func=_build_xg)

    ablate = commands.add_parser(
        "ablate", help="measure what each feature group adds out of sample"
    )
    ablate.add_argument("--games", default="data/raw/nhl_games.csv")
    ablate.add_argument("--minimum-training-games", type=int, default=300)
    ablate.add_argument("--forecast-kind", choices=("morning", "final"), default="final")
    ablate.add_argument("--ladder", help="comma-separated feature groups, in order")
    ablate.add_argument("--output", default="data/processed/ablation.csv")
    ablate.set_defaults(func=_ablate)

    verify = commands.add_parser("verify", help="run adversarial leakage checks on real data")
    verify.add_argument("--games", default="data/raw/games_with_xg.csv")
    verify.add_argument("--minimum-training-games", type=int, default=500)
    verify.set_defaults(func=_verify)

    publish = commands.add_parser("publish", help="build the dashboard's JSON snapshot")
    publish.add_argument("--backtest", default="data/processed/backtest_cutoff_2020_2025.csv")
    publish.add_argument("--games", default="data/raw/games_with_xg.csv")
    publish.add_argument("--ablation", default="data/processed/ablation_with_xg.csv")
    publish.add_argument("--upcoming", default="data/processed/forecasts.csv")
    publish.add_argument("--output", default="web/data/snapshot.json")
    publish.set_defaults(func=_publish)

    statuses = commands.add_parser(
        "capture-statuses", help="capture timestamped player availability statuses"
    )
    statuses.add_argument("--ledger", default="data/ledger")
    statuses.add_argument("--output", default="data/raw/player_statuses.csv")
    statuses.set_defaults(func=_capture_statuses)

    forecast = commands.add_parser(
        "forecast", help="generate versioned morning or final point-in-time forecasts"
    )
    forecast.add_argument("--history", required=True)
    forecast.add_argument("--schedule", required=True)
    forecast.add_argument("--forecast-kind", choices=("morning", "final"), required=True)
    forecast.add_argument("--ledger", default="data/ledger")
    forecast.add_argument("--output", default="data/processed/forecasts.csv")
    forecast.add_argument("--previous", help="earlier forecast CSV for change detection")
    forecast.add_argument("--changes-output", default="data/processed/forecast_changes.csv")
    forecast.add_argument("--change-threshold", type=float, default=0.05)
    forecast.set_defaults(func=_forecast)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
