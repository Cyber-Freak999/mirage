# BACKLOG: Mirage

**Version:** 3.1 | **Last Updated:** 2026-09-15
**Source:** product-spec.md, README.md, AGENTS.md, docs/superpowers/plans/2026-09-06-train-baseline-model.md, docs/superpowers/plans/2026-09-13-combined-baseline-training.md, full-stack code audit 2026-09-15 (`app/model/ensemble.py`, `app/retrain/*`, `app/schema/*`, `app/api/`, `app/honeypot/`, `dashboard/app.py`, `docker-compose.yml`, Dockerfiles, on-disk DB state)

Status: `[x]` done, `[ ]` not done, `[~]` partial, `[c]` blocked, `--` removed/monitor-only.

> **Single source of truth** for all pending project work. Phase order (agreed 2026-09-15): Tier 1 model honesty → Tier 2 adaptive loop → Tier 3 security & deploy → Tier 4 hygiene → Tier 5 release gates. Execute one phase at a time; write a detailed execution plan per phase before starting it.

---

## How to use this backlog

- **Tier** — priority band: `T1`-`T5` pending phases below; `T0` (training milestone) is complete and kept for history.
- **Component** -- which part it touches (`app`, `dashboard`, `infra`, `docs`, `tests`).
- **Priority** -- `must` / `should` / `could`.
- Items are ordered top-to-bottom; work proceeds in order, blocking dependencies noted inline.

---

## Master Priority Order (all pending, top-first)

### Tier 0 -- Finish the training milestone (COMPLETE 2026-09-15)

1. `[x]` **CICIDS `inf` -> 0 (TDD fix).** `app` | `must` — coerce `±inf -> NaN -> 0` on each sampled chunk (`app/schema/datasets.py`); all 7 CICIDS files load at 0.1. Commit: `fix: coerce inf to zero when loading CICIDS2017` (`4850502`).
2. `[x]` **Re-measure combined load + split.** `infra` | `must` — peak 678 MiB (693,784 KB) at `--cicids-sample 0.1 --unsw-sample 0.5`; no swap used; fits host budget.
3. `[x]` **Run combined baseline.** `app` | `must` — trained 1,187,060 rows (combined 1,483,826 rows, 0.1 CICIDS / 0.5 UNSW, k-folds 5, val 0.2); wall clock 1:08:25; promoted as `v1789430774`.
4. `[x]` **Verify artifacts + API.** `app` | `must` — `champion.pkl|json` (`v1789430774`), `v1789430774.pkl|json`, `validation_set.npz`; `validation.db` rows `promoted=1` (`v1788810565` -> `v1789324008` -> `v1789430774`); `POST /api/score` HTTP 200 with new `model_version` (SQLi payload scored attack 0.76).
5. `[x]` **Record baseline report.** `docs` | `must` — `docs/baseline-report.md` (val F1 0.5637 / precision 0.4013 / recall 0.9465 / AUC 0.9240, row counts, class distribution, training time, top-8 features).
6. `[x]` **Supersede placeholder champion.** `app` | `must` — `v1788810565` archived; live `champion` is `v1789430774`; API serves the new version.

### Tier 1 -- Model honesty (Phase A)

> Audit context: `method_get`/`method_post` carry 75% of feature importance, but the dataset mappers *fabricate* `method = "POST" if total_bytes > 1000 else "GET"` (`_cicids_row_to_http`/`_unsw_row_to_http`, `app/schema/datasets.py:389-453`) — the top features are artifacts of our own synthetic mapping, not attack semantics. `ModelVersion.validation_metrics` are computed on the training set itself (`app/model/ensemble.py:180-203`), and the 0.5 decision threshold is hardcoded in four places.

7. `[x]` **Truthful validation metrics.** `app` | `must` — `fit()` now stores `training_metrics`; `ChampionChallenger.validate()` writes held-out `validation_metrics` into model metadata (both promotion paths save via the gate). Old JSONs still load via defaults. Commit `2a54742`.
8. `[x]` **Threshold calibration.** `app` | `must` — single-source `decision_threshold` persisted in `ModelVersion`, consumed by `evaluate_model`, `predict`, both API score endpoints; `app/model/calibration.py` (`select_threshold`, `precision_recall_table`); gate calibrates challengers before comparing. Champion calibrated to 0.80 (F1 0.6399 vs 0.5633 at 0.5). Commits `f9d5969`.
9. `[x]` **Temporal-split evaluation.** `app`/`tests` | `must` — `scripts/evaluate.py --mode temporal`: train Mon/Tue → eval Thu/Fri gives AUC 0.644 vs 0.923 random split (same-period leakage quantified). Commit `51c382b`.
10. `[x]` **Cross-dataset evaluation.** `app` | `should` — `--mode cross`: CICIDS→UNSW AUC 0.180 (anti-correlated), UNSW→CICIDS 0.491 (random); the synthetic-HTTP mapping learns dataset-specific artifacts, not transferable semantics.
11. `[x]` **Shortcut-learning audit.** `app`/`docs` | `must` — `--mode zero-features`: AUC 0.923→0.796 without `method_*`, with `user_agent_entropy` (next artifact) absorbing 61% importance. Findings + PR table + calibrated threshold recorded in `docs/baseline-report.md`. Commit `6b1784a`.

### Tier 2 -- Adaptive loop (Phase B)

> Audit context: the loop (capture -> drift -> review -> retrain -> promote -> reload) is fully scaffolded but **has never run and cannot run**: the drift reference is never built from training data (the only caller passes a `np.zeros((1, 25))` placeholder, `scheduler.py:62-73`), `data/reference/` is empty, `drift.db`/`review.db`/`scheduler.db` do not exist on disk, and the review->retrain link has zero callers.

12. `[ ]` **Drift reference from training.** `app` | `must` — `train.py` must call `set_reference()` on real training data and persist to `data/reference/` so `check_drift` works. Also fix tracked-feature mapping: importance rank `i` is compared against column `i` positionally (`drift.py:137-139,194-201`), mislabeling feature names.
13. `[ ]` **Capture->features fidelity.** `app` | `must` — `_rows_to_features` (`scheduler.py:180-200`) JSON-parses urlencoded form bodies (always fails, so POST body features are empty), reads a `path` column that includes the query string, and passes only the User-Agent header (`header_count` always 1, `cookie_count` always 0). Same bug duplicated in `dashboard/app.py:567-591`. Store headers/body in the honeypot DB in a directly convertible form.
14. `[ ]` **Wire the review gate.** `app`/`dashboard` | `must` — `approve()/reject()` have no API endpoint or dashboard control; the 72h auto-proceed (`review_gate.py:199-225`) has no consumer; `drift_result_id=0` is hardcoded (`scheduler.py:168`); `process_review` — the only review->retrain path (`scheduler.py:202-242`) — has zero callers.
15. `[ ]` **Fix retrainer entrypoint + defaults.** `infra` | `must` — compose runs `python -m app.retrain.scheduler` (`docker-compose.yml:67`) but that module has no `__main__`, so the container restart-loops; the working entry `scheduler_entry.py` is dead code; its config defaults `cicids_dir`/`unsw_dir` to `None`, which would train on honeypot-only all-label-1 data.
16. `[ ]` **End-to-end loop test.** `tests` | `must` — synthetic capture burst -> drift trigger -> review approve -> retrain -> promotion -> API reload, plus unit tests for drift math and promote/reject logic (spec §16; note: §16 "Testing Strategy" *does* exist at `product-spec.md:152-156` — the old claim it was missing was stale). Supersedes old #10.

### Tier 3 -- Security & deploy (Phase C)

> Audit context: zero auth/CORS/rate-limiting/request-size limits anywhere; `SECRET_KEY` never set; `python-dotenv` declared but never called; the dashboard runs the Dash dev server with `debug=True` in Docker; both Dockerfiles run as root; healthchecks call `curl`, which is not installed in either image. The honeypot itself is fully simulated (no subprocess/eval/file writes), so containment risk is low today — re-check if that changes.

17. `[ ]` **API hardening.** `app` | `must` — API-key auth (finally use `python-dotenv`), `MAX_CONTENT_LENGTH`, stop returning raw exception strings (`api/__init__.py:97,137,180`), protect `/api/model/reload`, set `SECRET_KEY`.
18. `[ ]` **Dashboard production mode.** `dashboard` | `must` — no `debug=True` in prod (`dashboard/app.py:613`); auth on admin actions (Reload Model, Trigger Drift Check, Create Validation Set, `:511-521`); replace the random-placeholder "Model Confidence" tab (`:68-76`) with real data.
19. `[ ]` **Docker hardening.** `infra` | `must` — non-root `USER` in both Dockerfiles; replace curl-based healthchecks with a python one-liner (curl is not installed); consider splitting the single shared `mirage-internal` network so dashboard/api/honeypot/retrainer are not all mutually reachable; drop the obsolete compose `version:` key.
20. `[ ]` **Capture-data retention policy.** `docs`/`infra` | `should` — attacker payloads may contain PII; define retention (e.g. purge `raw_request` after N days, keep aggregated features).

### Tier 4 -- Hygiene (Phase D)

21. `[ ]` **CI.** `infra` | `must` — GitHub Actions workflow: ruff check + format + pytest on push/PR (`.github/` has no workflows; nothing runs tests today).
22. `[ ]` **mypy.** `infra` | `should` — type hints already exist; add mypy config + dependency and get it green.
23. `[ ]` **Model retention.** `infra` | `should` — `data/models/` grows ~26.5 MB per version, unbounded; spec §9 says never delete, so make pruning opt-in (keep champion + last N).
24. `[ ]` **SQLite backups.** `infra` | `should` — `honeypot.db`/`validation.db`/`drift.db` are single files with no backup story; a simple `.backup` script/cron.
25. `[ ]` **Docs & dead-code accuracy.** `docs`/`app` | `should` — README `uv run python -m app` is broken (no `app/__main__.py`, old #9); `requirements.txt` duplicates `pyproject.toml` (Dockerfile drift risk); remove or wire vestigial code: all-None `CICIDS_FEATURE_MAP`/`UNSW_FEATURE_MAP`, unused `DriftMonitor.reference_window`, never-written `feature_stats` table, always-0 `decoy_indicator`, double-save on promotion in `train.py`, `--random-state` flag not propagated to `train_initial_model`, `create_validation_set` silently overwriting the "fixed" validation npz, UNSW `DtypeWarning` (old #14).

### Tier 5 -- Release gates

26. `[ ]` **Add Apache-2.0 LICENSE** (full text) at repo root. `docs` | `must` — currently absent.
27. `[~]` **Attribution / NOTICE.** `docs` | `must` — CICIDS2017 + UNSW-NB15 citations and the UNSW academic-use restriction verified present (README L100-101); decide whether a NOTICE file is needed and add it if so.
28. `[ ]` **Tracking-docs fate + commit.** `infra` | `should` — decide whether `docs/superpowers/plans/*.md` remain repo artifacts; commit `BACKLOG.md` + `docs/` (or relocate) before release.
29. `[ ]` **Final gates + flip public.** `infra` | `must` — re-run `uv run ruff check .` + `uv run pytest tests/`, confirm `git status` clean, then flip repo visibility. Depends on Tier 3 completion.

**Verified this pass (2026-09-13, still true 2026-09-15):** gitignore covers `data/`, `*.db*`, `*.npz`, `*.joblib`, `*.bin`, `.env`; secrets scan clean (no `.env`, no tracked key/secret patterns); quality gates green (ruff clean, 44 tests passed). Model versioning works end-to-end; deterministic re-runs produce byte-identical models.

### Monitor-only

- `--` **"Benign" class frozen from public datasets.** `docs` | monitor — by design; retraining only sees captured honeypot (attack-only) + public data, so drift detection is the only adaptive lever on benigns. Revisit if a benign-capture source is added (e.g. mirroring real traffic to a twin service).

### Dropped / merged (v3.0 restructure)

- old #10 (integration test) -> superseded by #16 (end-to-end loop test; spec §16 exists).
- old #13 (weak real-request scoring) -> merged into #11 (shortcut audit).
- old #14 (UNSW DtypeWarning) -> merged into #25 (docs & dead-code accuracy).
- old #9 (AGENTS.md run command) -> merged into #25.
