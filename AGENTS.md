# AGENTS.md

## Project Overview

Mirage is a honeypot-driven adaptive intrusion detection system (IDS). A high-interaction honeypot lures and captures real attack traffic, which is combined with public benchmark datasets to train a stacking ensemble that detects intrusions in real time and retrains itself when detection starts to drift.

## Tech Stack

- Python 3.11, Flask, Plotly Dash, scikit-learn, XGBoost
- SQLite (WAL mode), APScheduler, Docker
- Package manager: uv
- Linter/formatter: ruff

## Development Setup

- `uv sync` to install dependencies
- `uv run python -m app` for local dev
- Docker: `docker compose up`

## Commands

- `uv run ruff check .` — lint
- `uv run ruff format .` — format
- `uv run pytest tests/` — run tests

## Architecture

Request flow: Honeypot → SQLite → Feature Extraction → Model → Dashboard

Key paths:
- `app/api/` — Flask API endpoints
- `app/honeypot/` — SQLi and RCE honeypot endpoints
- `app/model/` — Stacking ensemble (RF + XGBoost + LR)
- `app/retrain/` — Drift detection, review gate, champion/challenger
- `app/schema/` — Feature extraction and dataset loaders
- `dashboard/` — Plotly Dash monitoring dashboard

## Code Standards

- All public functions, classes, and methods must have docstrings
- Follow Google style docstrings
- Type hints required for function signatures
- ruff for linting and formatting (no other formatters)

## Known Issues

- Integration test needed (spec §16)
- "Benign" class comes only from public datasets, frozen at collection time

## Testing

- `uv run pytest tests/` for unit tests
- Integration test needed (spec §16)

## Commit Style

- Conventional commits or your preference
