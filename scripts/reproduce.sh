#!/usr/bin/env bash
# Rebuild the headline number from scratch.
#
#   ./scripts/reproduce.sh          recompute everything downstream of raw data (~2 min)
#   ./scripts/reproduce.sh --full   also re-download every game and shot from the NHL (~35 min)
#
# The point is that nothing here trusts a stored result. Features, the expected
# goals model, the backtest, and the leakage checks are all recomputed, and the
# --full path re-fetches the source data from the NHL's own API.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/nhl-predictor"
GAMES="data/raw/history_2020_2025.csv"
SHOTS="data/raw/shots_2020_2025.csv"
ENRICHED="data/raw/games_with_xg.csv"
FULL="${1:-}"

step() { printf '\n=== %s ===\n' "$1"; }

if [ "$FULL" = "--full" ]; then
  step "Re-downloading five seasons from the NHL API"
  rm -f "$GAMES" "$SHOTS" data/raw/*_v3.csv
  $PY collect --start 2021-01-13 --end 2021-05-19 --output data/raw/2020-21_v3.csv
  $PY collect --start 2021-10-12 --end 2022-04-29 --output data/raw/2021-22_v3.csv
  $PY collect --start 2022-10-07 --end 2023-04-14 --output data/raw/2022-23_v3.csv
  $PY collect --start 2023-10-10 --end 2024-04-18 --output data/raw/2023-24_v3.csv
  $PY collect --start 2024-10-04 --end 2025-04-18 --output data/raw/2024-25_v3.csv
  $PY combine data/raw/2020-21_v3.csv data/raw/2021-22_v3.csv data/raw/2022-23_v3.csv \
              data/raw/2023-24_v3.csv data/raw/2024-25_v3.csv --output "$GAMES"

  step "Re-downloading play-by-play shots (this is the slow part)"
  $PY collect-shots --games "$GAMES" --output "$SHOTS"
fi

step "Fitting expected goals on 2020-21 and 2021-22, then freezing it"
$PY build-xg --shots "$SHOTS" --games "$GAMES" --train-seasons 2020,2021 --output "$ENRICHED"

step "Walk-forward backtest"
$PY backtest --games "$ENRICHED" --minimum-training-games 500 \
  --predictions data/processed/backtest_reproduced.csv | tail -n 30

step "Adversarial leakage checks"
$PY verify --games "$ENRICHED" --minimum-training-games 500

step "Done"
echo "Compare the accuracy and log loss above against the README's claims."
