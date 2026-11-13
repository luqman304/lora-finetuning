.PHONY: install test lint data eval-zero eval-few train eval-lora report

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	HF_HUB_OFFLINE=1 .venv/bin/pytest

lint:
	.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests

data:
	.venv/bin/lora-ft prepare

eval-zero:
	.venv/bin/lora-ft eval --name zero-shot --prompt-style full

eval-few:
	.venv/bin/lora-ft eval --name few-shot-8 --prompt-style full --shots 8

train:
	.venv/bin/lora-ft train --out-dir outputs/lora

eval-lora:
	.venv/bin/lora-ft eval --name lora --adapter outputs/lora/adapter --prompt-style short

report:
	.venv/bin/lora-ft report --names zero-shot,few-shot-8,lora --baseline few-shot-8 --out results/RESULTS.md
