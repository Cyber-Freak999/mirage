# Combined Baseline Report

**Date:** 2026-09-15 (honesty audit appended same day; flow-mapping retrain appended 2026-09-21)
**Model:** stacking ensemble (RF + XGBoost + LR meta-learner), k-fold cross-validated
**Run:** `python -m app.model.train` — CICIDS2017 sample 0.1, UNSW-NB15 sample 0.5, k-folds 5, val-size 0.2, random_state 42
**Source commit (loader fix):** `4850502` · **Honesty audit commit:** see git log (`phase-a/model-honesty`)

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

> **Correction (honesty audit):** the original version of this table quoted fit-time
> self-scored metrics (the `fit()` bug fixed in this phase). The numbers below are the
> true held-out values from the champion/challenger gate (`validation.db` row 3),
> evaluated at the then-default 0.5 threshold.

| Metric | Value @ t=0.5 |
|---|---|
| Precision | 0.4015 |
| Recall | 0.9441 |
| F1 | 0.5633 |
| AUC | 0.9233 |

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
- `POST /api/score` (SQLi payload `pass=1 OR 1=1`) → HTTP 200, score 0.761, `model_version":"v1789430774"`. Label was `"attack"` at the pre-audit 0.5 threshold; at the calibrated 0.80 threshold it reads `"benign"` (0.76 < 0.80) — see the honesty audit below.
- `GET /api/model/info` → `"version":"v1789430774"`, `training_samples: 1187060`, held-out `validation_metrics` (post-fix, now truthful)

## Model honesty audit (2026-09-15, phase A)

Three studies (`scripts/evaluate.py`) + threshold calibration, run against the champion `v1789430774` and fresh models trained on the same loaders.

### 1. Threshold calibration (champion, held-out set)

The 0.5 decision threshold was hardcoded; the champion is now calibrated to the F1-optimal **0.80** (stored in `champion.json`, consumed by `evaluate_model`, `predict`, and both API score endpoints):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (old default) | 0.4015 | 0.9441 | 0.5633 |
| **0.80 (deployed, F1-optimal)** | **0.5313** | 0.8043 | **0.6399** |
| 0.90 (precision floor ≥ 0.6) | 0.8826 | 0.1506 | 0.2572 |

The score distribution has a sharp cliff between 0.85 and 0.90 (the `precision_floor` objective only reaches P ≥ 0.6 by sacrificing 85% of recall). A recall-biased deployment can lower the threshold — the mechanism now exists; the tradeoff is explicit above.

### 2. Temporal-split study (train Mon/Tue → eval Thu/Fri, CICIDS-only, 0.1 sample)

| | Rows | Attacks |
|---|---|---|
| Train (Mon+Tue) | 97,583 | 1,406 (1.4%) |
| Eval (Thu+Fri) | 116,221 | 29,170 (25.1%) |

**AUC 0.644** (vs 0.923 on the random split); F1 0.44 at 0.5, 0.50 at the calibrated 0.10. The random-split validation set was leaking same-period information: trained on two early days, the model barely generalizes to the attack types of later days. Feature importance also inverts — `method_get` collapses from 0.42 to 0.06 and path/param features dominate.

### 3. Cross-dataset study (train one dataset, eval the other)

| Direction | AUC | F1 @0.5 | Calibrated F1 |
|---|---|---|---|
| CICIDS → UNSW | **0.180** (anti-correlated) | 0.0001 | 0.2303 |
| UNSW → CICIDS | **0.491** (random) | 0.1901 | 0.2401 |

The synthetic-HTTP mapping learns **dataset-specific artifacts, not transferable attack semantics**: the two dataset mappers fabricate HTTP methods with opposite biases (UNSW-trained model puts 51% importance on `method_get` and treats GET as the attack signal; the CICIDS mapping leans the other way), so each model is useless — or anti-useful — on the other dataset.

### 4. Feature-ablation study (method_* zeroed, combined 0.1/0.5, same 80/20 split)

Zeroing `method_get`, `method_post`, `method_other` in both train and eval:

| Model | AUC | F1 @0.5 | Top feature |
|---|---|---|---|
| Champion (all features) | 0.9233 | 0.5633 | method_get (0.42) |
| **method_* zeroed** | **0.7961** | 0.4119 | **user_agent_entropy (0.61)** |

`method_*` carries roughly a third of the ranking quality — and the moment it is removed, `user_agent_entropy` (another fabricated mapping artifact) absorbs 61% of importance, confirming the ensemble latches onto synthetic-mapping artifacts in a chain rather than attack semantics.

### Findings

1. **The 0.92 AUC headline overstates the model.** Honest estimates: ~0.64 across time, ~0.18-0.49 across datasets. The random-split number measures memorization of dataset artifacts, including our own `method ~ bytes>1000` fabrication.
2. **Recall-heavy 0.5 default was hiding a 60% false-alarm rate**; deployed threshold is now 0.80 with an explicit tradeoff table.
3. **Real-payload scoring stays weak** (SQLi probe scores 0.76; borderline at the new threshold) — consistent with the mapping-artifact finding. Fixing this means better feature grounding (Tier 2 capture fidelity work), not more training data.

## Retrain on direct flow-to-feature mapping (2026-09-21)

Precursor fix (`eb2e9bc`): `_cicids_row_to_http` / `_unsw_row_to_http` fabricated
`method` from arbitrary flow thresholds (`total_bytes > 1000`, `Spkts > Dpkts`),
giving `method_get`/`method_post` 75% of feature importance. Replaced with direct
flow-column → 25-feature mappers (`_cicids_frame_to_features`,
`_unsw_frame_to_features`); `method_*` is now constant 0.0 for public data
(no true HTTP method exists in flow records). Retrained from scratch:

**Run:** `python -m app.model.train --cicids-sample 0.05 --unsw-sample 0.25`
(k-folds 5, val-size 0.2, random_state 42). Sample fractions halved vs the 0.1/0.5
baseline: two full-size attempts were both OOM-killed at the "Retraining base
learners on full dataset" stage (host: 3.8 GiB RAM + 1 GiB swap, both under
pressure — journal logged `Under memory pressure, flushing caches`). Model config
unchanged, so the comparison stays honest; only data volume differs.

| Source | Rows loaded | Attacks |
|---|---|---|
| CICIDS2017 (0.05) | 106,901 | 15,262 |
| UNSW-NB15 (0.25) | 635,011 | 80,164 |
| **Combined** | **741,912** | **95,426** |

| Split | Rows | Attacks |
|---|---|---|
| Train (80%) | 593,529 | 76,341 |
| Validation (20%, persisted `data/validation_set.npz`) | 148,383 | 19,085 |

**Validation (held-out 20%, champion/challenger gate, calibrated threshold 0.95):**

| Metric | Value |
|---|---|
| Precision | 0.9594 |
| Recall | 0.9824 |
| F1 | 0.9708 |
| AUC | 0.9997 |

Challenger `v1789976193` vs prior champion `v1789430774`: **promoted**
(`Challenger beats champion: F1 0.9708 > 0.0024`). The old champion collapses to
F1 0.0024 on the new validation set — it was scoring the `method_*` artifact,
which is now all zeros. This is the strongest confirmation of the audit finding:
the previous model learned our fabrication, not attacks.

**Top-8 features by importance (no `method_*`, no `user_agent_entropy`):**

1. header_count — 0.5576
2. path_length — 0.1467
3. payload_length — 0.0773
4. longest_param_len — 0.0581
5. avg_param_len — 0.0532
6. param_name_entropy — 0.0200
7. param_value_entropy — 0.0185
8. digit_ratio — 0.0152

**Training time:** ~30 min wall clock (2026-09-21T07:06:22Z → 07:36:54Z).

**API verification:** `GET /api/health` → healthy, `model_version":"v1789976193"`;
`GET /api/model/info` → 593,529 training samples, held-out metrics as above.

**Known regression (documented, not a bug):** the SQLi probe
(`pass=1 OR 1=1`) that scored 0.76 under the old champion now scores **0.003
(benign)**. Attack-keyword features (`sql_keyword_count`, `rce_keyword_count`,
…) are identically 0 across all public flow data, so the retrained model assigns
them no weight. Real-HTTP attack detection now depends on honeypot capture data
entering retraining — i.e. the Tier 2 adaptive loop. Until the loop runs, the
model is a flow-anomaly classifier, not a payload-signature detector. The
temporal-split (AUC 0.64) and cross-dataset (AUC 0.18–0.49) caveats from the
audit still stand; re-running `scripts/evaluate.py` on the new pipeline is open.
