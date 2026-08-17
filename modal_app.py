"""Deploy with: modal deploy modal_app.py.

Before deployment, upload a historical completed-game CSV as
``raw/history.csv`` to the ``nhl-predictor-data`` Modal Volume.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import pandas as pd

app = modal.App("nhl-predictor")
data_volume = modal.Volume.from_name("nhl-predictor-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pandas", "requests", "scikit-learn")
    .add_local_python_source("nhl_predictor", copy=True)
)

DATA_ROOT = Path("/data")
EASTERN = ZoneInfo("America/New_York")


def _today_eastern() -> date:
    return datetime.now(UTC).astimezone(EASTERN).date()


def _history_path() -> Path:
    return DATA_ROOT / "raw" / "history.csv"


def _refresh_completed_history(today: date) -> pd.DataFrame:
    """Merge a rolling three-day final-results window into persisted history."""

    from nhl_predictor.nhl_api import collect_schedule

    history_path = _history_path()
    if not history_path.exists():
        raise FileNotFoundError(
            "Seed /data/raw/history.csv before deploying; it must contain completed historical games."
        )
    history = pd.read_csv(history_path)
    recent = collect_schedule(today - timedelta(days=3), today - timedelta(days=1))
    merged = pd.concat([history, recent], ignore_index=True)
    merged = merged.drop_duplicates("game_id", keep="last").sort_values("start_time_utc")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False)
    return merged


def _score_free_schedule(game_date: date) -> pd.DataFrame:
    from nhl_predictor.nhl_api import fetch_schedule_day

    schedule = fetch_schedule_day(game_date)
    schedule = schedule[schedule["game_date"] == game_date.isoformat()].copy()
    schedule["home_score"] = pd.NA
    schedule["away_score"] = pd.NA
    return schedule


def _write_forecast(forecast: pd.DataFrame, forecast_kind: str, game_date: date) -> Path:
    output = DATA_ROOT / "processed" / f"{game_date.isoformat()}_{forecast_kind}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output, index=False)
    return output


@app.function(
    image=image,
    volumes={"/data": data_volume},
    schedule=modal.Cron("0 9 * * *", timezone="America/New_York"),
    timeout=900,
)
def morning_forecast() -> dict[str, object]:
    """Capture official sources and create the early daily forecast."""

    from nhl_predictor.ledger import ObservationStore
    from nhl_predictor.nhl_api import capture_official_observations
    from nhl_predictor.workflow import generate_forecasts

    data_volume.reload()
    today = _today_eastern()
    store = ObservationStore(DATA_ROOT / "ledger")
    capture_official_observations(today, store, include_injury_reports=True)
    history = _refresh_completed_history(today)
    forecast = generate_forecasts(history, _score_free_schedule(today), store, forecast_kind="morning")
    output = _write_forecast(forecast, "morning", today)
    data_volume.commit()
    return {"forecast_count": len(forecast), "output": str(output)}


@app.function(
    image=image,
    volumes={"/data": data_volume},
    schedule=modal.Cron("*/15 * * * *"),
    timeout=900,
)
def final_forecast_refresh() -> dict[str, object]:
    """Refresh only games whose final cutoff is due in the next 15 minutes."""

    from nhl_predictor.cutoffs import with_forecast_cutoff
    from nhl_predictor.ledger import ObservationStore
    from nhl_predictor.nhl_api import capture_official_observations
    from nhl_predictor.workflow import generate_forecasts, material_prediction_changes

    data_volume.reload()
    today = _today_eastern()
    now = pd.Timestamp.now(tz="UTC")
    store = ObservationStore(DATA_ROOT / "ledger")
    capture_official_observations(today, store, include_injury_reports=False)
    history = _refresh_completed_history(today)
    schedule = _score_free_schedule(today)
    scheduled_with_cutoffs = with_forecast_cutoff(schedule, "final")
    due = scheduled_with_cutoffs[
        scheduled_with_cutoffs["prediction_cutoff_utc"].between(
            now, now + pd.Timedelta(minutes=15), inclusive="left"
        )
    ]
    if due.empty:
        return {"forecast_count": 0, "reason": "no final forecasts due"}

    final = generate_forecasts(history, due, store, forecast_kind="final")
    final_path = _write_forecast(final, "final", today)
    morning_path = DATA_ROOT / "processed" / f"{today.isoformat()}_morning.csv"
    changes = (
        material_prediction_changes(pd.read_csv(morning_path), final)
        if morning_path.exists()
        else pd.DataFrame()
    )
    notification_path = DATA_ROOT / "notifications" / f"{today.isoformat()}_final_changes.csv"
    notification_path.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(notification_path, index=False)
    data_volume.commit()
    return {
        "forecast_count": len(final),
        "changed_predictions": len(changes),
        "forecast_output": str(final_path),
        "notification_output": str(notification_path),
    }
