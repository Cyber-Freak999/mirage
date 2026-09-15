# Combined Baseline Report

**Date:** 2026-09-15
**Model:** stacking ensemble (RF + XGBoost + LR meta-learner), k-fold cross-validated
**Run:** `python -m app.model.train` — CICIDS2017 sample 0.1, UNSW-NB15 sample 0.5, k-folds 5, val-size 0.2, random_state 42
**Source commit (loader fix):** `4850502`

## Environment

- CPU: 4 cores
- RAM: 3.8 GiB total; load+split peak 678 MiB (measured, Task 2); training peak ~873 MiB RSS observed (estimate was 1.2-1.5 GB — actual came in lower); no swap used
- API run command: `uv run flask --app "app:create_app()" run --port 5000`

## Data

| Source | Rows loaded | Attacks |
|---|---|---|
| CICIDS2017 (0.1) | 213,804 | 30,574 |
| UNSW-NB15 (0.5) | 1,270,022 | 160,583 |
| **Combined** | **1,483,826** | **191,157** |

| Split | Rows | Attacks |
|---|---|---|
| Train (80%) | 1,187,060 | 152,925 |
| Validation (20%, persisted `data/validation_set.npz`) | 296,766 | 38,232 |

## Validation (held-out 20%, via champion/challenger gate)

| Metric | Value |
|---|---|
| Precision | 0.4013 |
| Recall | 0.9465 |
| F1 | 0.5637 |
| AUC | 0.9240 |

Challenger `v1789430774` vs prior champion `v1789324008`: **promoted** (`Challenger beats champion: F1 0.5633 > 0.5633` — exact tie at display precision; the gate promotes on `>=`). Champion lineage: placeholder `v1788810565` (10K smoke slice, F1 0.5536) → `v1789324008` (2026-09-13 run) → `v1789430774` (this run). The two real-data runs are byte-identical deterministic twins (same seed, same data), confirming `random_state=42` reproducibility. The placeholder `v1788810565` artifacts remain archived in `data/models/`.

Note: precision/recall asymmetry reflects `scale_pos_weight=6.76` (class imbalance 1,034,135 benign vs 152,925 attacks) — the model is deliberately recall-biased; AUC 0.9240 shows strong ranking quality.

## Training time

- Wall clock: 1:08:25 (2026-09-14T22:59:33Z → 2026-09-15T00:07:58Z)
  - Loading + split: ~32 min; 5 CV folds: ~28 min; full-data retrain + validation: ~7 min

## Top-8 features by importance

1. method_get — 0.4220
2. method_post — 0.3263
3. user_agent_entropy — 0.0957
4. path_length — 0.0216
5. path_depth — 0.0188
6. digit_ratio — 0.0186
7. longest_param_len — 0.0176
8. special_char_density — 0.0130

## API verification

- `GET /api/health` → `{"status":"healthy","model_version":"v1789430774"}`
- `POST /api/score` (SQLi payload `pass=1 OR 1=1`) → HTTP 200, `"label":"attack"`, score 0.761, `model_version":"v1789430774"`
- `GET /api/model/info` → `"version":"v1789430774"`, `training_samples: 1187060`
