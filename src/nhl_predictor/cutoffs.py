"""Prediction cutoff policies for historical replay and daily forecasts."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

FINAL_LEAD = pd.Timedelta(minutes=30)
EASTERN = ZoneInfo("America/New_York")


def with_forecast_cutoff(games: pd.DataFrame, forecast_kind: str = "final") -> pd.DataFrame:
    """Attach a point-in-time cutoff to every scheduled game.

    ``morning`` is 10:00 AM Eastern on the NHL game date. ``final`` is thirty
    minutes before puck drop. The source file may provide an explicit cutoff
    for a reproducible historical experiment.
    """

    if forecast_kind not in {"morning", "final"}:
        raise ValueError("forecast_kind must be 'morning' or 'final'")
    frame = games.copy()
    start = pd.to_datetime(frame["start_time_utc"], utc=True)
    if forecast_kind == "final":
        frame["prediction_cutoff_utc"] = start - FINAL_LEAD
    else:
        if "game_date" in frame.columns:
            local_date = pd.to_datetime(frame["game_date"]).dt.date
        else:
            local_date = start.dt.tz_convert(EASTERN).dt.date
        frame["prediction_cutoff_utc"] = [
            pd.Timestamp(day, tz=EASTERN).replace(hour=10).tz_convert("UTC") for day in local_date
        ]
    frame["forecast_kind"] = forecast_kind
    return frame
