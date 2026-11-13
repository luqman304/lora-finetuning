#!/usr/bin/env bash
# Reproduces every number in the README: fine-tune, then evaluate all three systems on the test split.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs
run() { echo "$(date '+%F %T') start $1"; shift; "$@"; }
[ -f data/processed/test.jsonl ] || .venv/bin/lora-ft prepare
run train .venv/bin/lora-ft train --out-dir outputs/lora > outputs/logs/train.log 2>&1
run eval-lora .venv/bin/lora-ft eval --name lora --adapter outputs/lora/adapter --prompt-style short > outputs/logs/eval-lora.log 2>&1
run eval-zero .venv/bin/lora-ft eval --name zero-shot --prompt-style full > outputs/logs/eval-zero.log 2>&1
run eval-few .venv/bin/lora-ft eval --name few-shot-8 --prompt-style full --shots 8 > outputs/logs/eval-few.log 2>&1
cp outputs/lora/train_summary.json results/train_summary.json
.venv/bin/lora-ft report --names zero-shot,few-shot-8,lora --baseline few-shot-8 --out results/RESULTS.md
echo "$(date '+%F %T') all done"
