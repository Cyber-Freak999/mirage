# Mirage — Adaptive Intrusion Detection System

Mirage is a honeypot-driven adaptive intrusion detection system (IDS). A high-interaction honeypot lures and captures real attack traffic, which is combined with public benchmark datasets (CICIDS2017, UNSW-NB15) to train a stacking ensemble that detects intrusions in real time. The system monitors its own detection performance and retrains itself when it starts to drift, protected by a review gate and a champion/challenger validation gate.

**This is a defensive research project. It is not built to exploit anyone.**

## How It Works

```
Honeypot → SQLite → Feature Extraction → Model → Dashboard
```

1. **Honeypot** — simulated, high-interaction web endpoints (SQLi-style login/search, RCE-style upload/admin) lure and log real attack traffic. Nothing is genuinely vulnerable.
2. **Feature extraction** — every request is mapped to a unified 25-feature HTTP-layer schema, shared with the public datasets so all sources train the model in the same feature space.
3. **Model** — a stacking ensemble: Random Forest + XGBoost base learners with a Logistic Regression meta-learner, trained with k-fold cross-validated stacking to prevent leakage.
4. **Retraining loop** — a PSI-based drift monitor triggers retraining when the incoming traffic distribution shifts; a human review queue (72h timeout) sanity-checks the data, and the retrained challenger only reaches production if it beats the champion on a fixed validation set.
5. **Dashboard** — a Plotly Dash UI showing the live attack feed, model confidence, feature drift, retraining history, and attack-type breakdown.

## Tech Stack

| Layer | Choice |
|---|---|
| Honeypot + API | Flask, gunicorn |
| Storage | SQLite (WAL mode) |
| Base models | Random Forest, XGBoost |
| Meta-learner | Logistic Regression (scikit-learn) |
| Scheduling | APScheduler (persistent job store) |
| Dashboard | Plotly Dash |
| Containerization | Docker (isolated, no outbound internet) |
| Public datasets | CICIDS2017, UNSW-NB15 |
| Package manager | uv |
| Lint / format | ruff |

## Getting Started

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)

### Local development

```bash
uv sync
uv run python -m app
```

The API and honeypot endpoints run on `http://localhost:5000`.

### Dashboard

```bash
uv run python -m dashboard.app
```

Dashboard runs on `http://localhost:8050`.

### Docker

```bash
docker compose up
```

This starts three isolated services — `honeypot-api`, `dashboard`, and `retrainer` — on an internal-only network with no outbound internet access, CPU/memory limits, and read-only filesystems. Service data (SQLite databases, model versions, drift references) persists in the `mirage-data` volume.

### Running tests

```bash
uv run pytest tests/
```

### Capture-data retention

Raw attacker payloads (`raw_request`, `headers_json`) may contain PII and are
purged after 30 days; structured fields and all training artifacts are kept:

```bash
uv run python scripts/purge_captures.py --days 30          # run the purge
uv run python scripts/purge_captures.py --days 30 --dry-run  # report only
```

Run this on a schedule (e.g. host cron) wherever captures accumulate.

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
app/
  api/           Flask API endpoints (/api/health, /api/score, /api/model/*)
  honeypot/      SQLi and RCE honeypot endpoints + SQLite logging
  model/         Stacking ensemble, model versioning, champion save/load
  retrain/       Drift monitor, review gate, champion/challenger, scheduler
  schema/        25-feature HTTP schema extractor + dataset loaders
dashboard/       Plotly Dash monitoring dashboard
tests/           Unit tests
```

## Known Limitations

- **Benign-class staleness** — all benign training examples come from the public datasets (honeypot traffic is 100% attack by design), so the model's notion of "benign" is frozen to when those datasets were collected. Only the attack side adapts over time.
- **Adversarial evasion** — no adversarial training is performed. A motivated attacker who can observe the model's features may eventually shape payloads to evade detection.
- **Drift-trigger blind spots** — drift monitoring tracks only the top 5-8 features by importance; drift in untracked features, or degradation visible only in prediction confidence, won't fire the trigger.

## Datasets

- **CICIDS2017** — Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
- **UNSW-NB15** — Moustafa and Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," 2015. UNSW-NB15 is made available for **academic research**; commercial use is restricted.

## Training the baseline model

Download the public benchmark CSVs into `data/raw/cicids2017/` and `data/raw/unsw-nb15/` (CICIDS2017 and UNSW-NB15; see the sources above — CICIDS2017 is form-gated and large, UNSW-NB15 is directly downloadable). The `scripts/fetch_csv.py` helper can grab individual CSVs into the right layout:

    uv run python scripts/fetch_csv.py --url <csv-url> --dest data/raw/unsw-nb15/part.csv

Then train the baseline champion:

    uv run python -m app.model.train

Flags control per-source sampling (`--cicids-sample`, `--unsw-sample` — default 0.1/0.5 to bound runtime), stacking folds (`--k-folds`), and validation holdout (`--val-size`). The run:

1. Loads + samples both datasets, mapping them to the unified 25-feature schema
2. Holds out a stratified 20% validation set (persisted to `data/validation_set.npz`)
3. Trains the stacking ensemble (RF + XGBoost + LR meta-learner) on the rest
4. Runs champion/challenger validation on the held-out set (no champion exists yet, so the first run always promotes)
5. On promotion, saves the versioned model and `champion.pkl` under `data/models/`

Artifacts live under gitignored `data/`, so nothing gets committed.

## Legal & Ethics

Mirage is a defensive honeypot for security research. It logs traffic (including source IPs) and uses captured data for model training. Public deployment should include a visible notice (e.g., `security.txt`) disclosing the honeypot's nature, log retention practices, and research-only usage. This project does not constitute legal advice; consult someone knowledgeable about local law (e.g., Nigeria's Cybercrimes Act 2015) before public deployment.
