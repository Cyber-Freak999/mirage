# BACKLOG: Mirage

**Version:** 3.4 | **Last Updated:** 2026-09-22
**Source:** product-spec.md, README.md, AGENTS.md, full-stack code audit 2026-09-15 (`app/model/ensemble.py`, `app/retrain/*`, `app/schema/*`, `app/api/`, `app/honeypot/`, `dashboard/app.py`, `docker-compose.yml`, Dockerfiles, on-disk DB state). (Session plan docs under `docs/superpowers/plans/` are git-ignored working notes, not repo artifacts — see #28.)

Status: `[x]` done, `[ ]` not done, `[~]` partial, `[c]` blocked, `--` removed/monitor-only.

> **Single source of truth** for project work. Tiers 0–5 are complete (2026-09-22); the remaining list is "Deferred — post-release follow-ups" below. Phase order was Tier 1 model honesty → Tier 2 adaptive loop → Tier 3 security & deploy → Tier 4 hygiene → Tier 5 release gates.

---

## How to use this backlog

- **Tier** — priority band: `T1`-`T5` pending phases below; `T0` (training milestone) is complete and kept for history.
- **Component** -- which part it touches (`app`, `dashboard`, `infra`, `docs`, `tests`).
- **Priority** -- `must` / `should` / `could`.
- Items are ordered top-to-bottom; work proceeds in order, blocking dependencies noted inline.

---

## Record (Tiers 0–5 complete; Deferred follow-ups below)

### Tier 0 -- Finish the training milestone (COMPLETE 2026-09-15)

> Historical note: items 3–6 name champion `v1789430774` — superseded 2026-09-21 by `v1789976193` (see Tier 1b). Lineage: `v1788810565` → `v1789324008` → `v1789430774` → `v1789976193` (live).

1. `[x]` **CICIDS `inf` -> 0 (TDD fix).** `app` | `must` — coerce `±inf -> NaN -> 0` on each sampled chunk (`app/schema/datasets.py`); all 7 CICIDS files load at 0.1. Commit: `fix: coerce inf to zero when loading CICIDS2017` (`4850502`).
2. `[x]` **Re-measure combined load + split.** `infra` | `must` — peak 678 MiB (693,784 KB) at `--cicids-sample 0.1 --unsw-sample 0.5`; no swap used; fits host budget.
3. `[x]` **Run combined baseline.** `app` | `must` — trained 1,187,060 rows (combined 1,483,826 rows, 0.1 CICIDS / 0.5 UNSW, k-folds 5, val 0.2); wall clock 1:08:25; promoted as `v1789430774`.
4. `[x]` **Verify artifacts + API.** `app` | `must` — `champion.pkl|json` (`v1789430774`), `v1789430774.pkl|json`, `validation_set.npz`; `validation.db` rows `promoted=1` (`v1788810565` -> `v1789324008` -> `v1789430774`); `POST /api/score` HTTP 200 with new `model_version` (SQLi payload scored attack 0.76).
5. `[x]` **Record baseline report.** `docs` | `must` — `docs/baseline-report.md` (val F1 0.5637 / precision 0.4013 / recall 0.9465 / AUC 0.9240, row counts, class distribution, training time, top-8 features).
6. `[x]` **Supersede placeholder champion.** `app` | `must` — `v1788810565` archived; live `champion` is `v1789430774`; API serves the new version.

### Tier 1 -- Model honesty (Phase A)

> Audit snapshot 2026-09-15 (pre-fix; resolved by items 7–11 below, superseded further by the flow-mapping precursor): `method_get`/`method_post` carried 75% of feature importance, but the dataset mappers *fabricated* `method = "POST" if total_bytes > 1000 else "GET"` (`_cicids_row_to_http`/`_unsw_row_to_http`, then `app/schema/datasets.py:389-453`) — the top features were artifacts of our own synthetic mapping, not attack semantics. `ModelVersion.validation_metrics` were computed on the training set itself (then `app/model/ensemble.py:180-203`), and the 0.5 decision threshold was hardcoded in four places.

7. `[x]` **Truthful validation metrics.** `app` | `must` — `fit()` now stores `training_metrics`; `ChampionChallenger.validate()` writes held-out `validation_metrics` into model metadata (both promotion paths save via the gate). Old JSONs still load via defaults. Commit `2a54742`.
8. `[x]` **Threshold calibration.** `app` | `must` — single-source `decision_threshold` persisted in `ModelVersion`, consumed by `evaluate_model`, `predict`, both API score endpoints; `app/model/calibration.py` (`select_threshold`, `precision_recall_table`); gate calibrates challengers before comparing. Champion calibrated to 0.80 (F1 0.6399 vs 0.5633 at 0.5). Commits `f9d5969`.
9. `[x]` **Temporal-split evaluation.** `app`/`tests` | `must` — `scripts/evaluate.py --mode temporal`: train Mon/Tue → eval Thu/Fri gives AUC 0.644 vs 0.923 random split (same-period leakage quantified). Commit `51c382b`.
10. `[x]` **Cross-dataset evaluation.** `app` | `should` — `--mode cross`: CICIDS→UNSW AUC 0.180 (anti-correlated), UNSW→CICIDS 0.491 (random); the synthetic-HTTP mapping learns dataset-specific artifacts, not transferable semantics.
11. `[x]` **Shortcut-learning audit.** `app`/`docs` | `must` — `--mode zero-features`: AUC 0.923→0.796 without `method_*`, with `user_agent_entropy` (next artifact) absorbing 61% importance. Findings + PR table + calibrated threshold recorded in `docs/baseline-report.md`. Commit `6b1784a`.

### Tier 1b -- Flow-mapping precursor (2026-09-21, unblocks honest retraining)

> Follow-up finding: Phase A measured the synthetic-method artifact but left the fabrication in place. Fixed before any further training.

- `[x]` **Direct flow-to-feature mapping.** `app` | `must` — replaced `_cicids_row_to_http`/`_unsw_row_to_http` with `_cicids_frame_to_features`/`_unsw_frame_to_features` (`app/schema/capture.py` pattern: shared converter, `app/schema/datasets.py`); `method_*` constant 0.0 for flow data (no true HTTP method exists in flow records). Commit: `feat: replace synthetic HTTP mapping with direct flow-to-feature extraction` (`eb2e9bc`).
- `[x]` **Retrain on fixed pipeline.** `app`/`docs` | `must` — two full-size (0.1/0.5) attempts OOM-killed at the full-data retrain stage on the 3.8 GiB host; succeeded at 0.05/0.25 (741,912 rows, ~30 min). Champion `v1789976193` promoted (held-out F1 0.9708 / precision 0.9594 / recall 0.9824 / AUC 0.9997 @ calibrated 0.95; top features all flow-derived, no `method_*`). Prior champion `v1789430774` scores F1 0.0024 on the new validation set — confirming it had learned the fabrication. Known regression (documented, by design): payload-keyword features are all-zero in flow data, so real SQLi probes score benign until honeypot captures arrive via the loop. Recorded in `docs/baseline-report.md`. Commit: `docs: record flow-mapping retrain in baseline report` (`888b273`).

### Tier 2 -- Adaptive loop (Phase B)

> Audit snapshot 2026-09-15 (pre-fix; resolved by items 12–16 below): the loop (capture -> drift -> review -> retrain -> promote -> reload) was fully scaffolded but **had never run and could not run**: the drift reference was never built from training data (the only caller passed a `np.zeros((1, 25))` placeholder, then `scheduler.py:62-73`), `data/reference/` was empty, `drift.db`/`review.db`/`scheduler.db` did not exist on disk, and the review->retrain link had zero callers.

12. `[x]` **Drift reference from training.** `app` | `must` — `train.py` persists `set_reference()` on promotion to `data/reference/`; `set_reference`/`check_drift` resolve columns by feature name, not importance-rank position (`drift.py`). Scheduler loads from disk instead of the `zeros((1, 25))` placeholder. Commit: `fix: build drift reference from training data with name-based columns`.
13. `[x]` **Capture->features fidelity.** `app` | `must` — honeypot logs `request.path` (query split out) + new `headers_json`/`content_type` columns (migrated via `ALTER TABLE`); single shared `rows_to_features` (`app/schema/capture.py`) replaces both buggy copies (scheduler + dashboard). Verified live: POST SQLi body captured with keyword hits. Commit: `fix: faithful capture-to-features conversion with shared converter`.
14. `[x]` **Wire the review gate.** `app`/`dashboard` | `must` — `POST /api/reviews/<id>/approve|reject` (404/409 semantics); dashboard Approve/Reject buttons; scheduler sweep consumes approved/auto-proceeded reviews into `process_review` each tick; real drift-check ids flow into reviews; new `DONE` status stops repeat retraining. Commit: `feat: wire review gate to API, dashboard, and scheduler sweep`.
15. `[x]` **Fix retrainer entrypoint + defaults.** `infra` | `must` — compose runs `python -m app.retrain.scheduler_entry` (old target had no `__main__`); entry defaults to `data/raw/*` dirs with loud fallback warnings; `RetrainingConfig` sample caps 0.05/0.25 (full-size OOMs; proven 2026-09-21) forwarded through `run_full_retrain`. Commit: `fix: working retrainer entrypoint with OOM-safe defaults`.
16. `[x]` **End-to-end loop test.** `tests` | `must` — `tests/test_loop_e2e.py`: synthetic burst -> drift trigger -> review approve -> retrain -> promotion -> API reload (19s, fully isolated tmp dirs incl. `REFERENCE_DIR`), plus PSI fire/quiet and promote tie/loss unit tests (spec §16). Commit: `test: end-to-end adaptive loop coverage`.

### Tier 3 -- Security & deploy (Phase C)

> Audit snapshot 2026-09-15 (pre-fix; resolved by items 17–20 below): zero auth/CORS/rate-limiting/request-size limits anywhere; `SECRET_KEY` never set; `python-dotenv` declared but never called; the dashboard ran the Dash dev server with `debug=True` in Docker; both Dockerfiles ran as root; healthchecks called `curl`, which is not installed in either image. The honeypot itself is fully simulated (no subprocess/eval/file writes), so containment risk was low — re-check if that changes.

17. `[x]` **API hardening.** `app` | `must` — `require_api_key` (fail-closed 503 unconfigured, 401 mismatch) on `/api/model/reload` + review endpoints; `SECRET_KEY`/`MAX_CONTENT_LENGTH` via dotenv in factory; generic error messages (no `str(e)` leaks); per-IP sliding-window limiter (60/min, env-tunable, logs-first per spec §13); `POST /api/score` logs to `score_log` for the confidence tab. Commit: `feat: harden API with key auth, limits, and error hygiene`.
18. `[x]` **Dashboard production mode.** `dashboard` | `must` — `debug` from `DASH_DEBUG` (default off); session admin gate (`check_admin_key` vs `MIRAGE_ADMIN_KEY`, fail closed) guarding System Actions + review decisions; confidence tab reads real `score_log` with empty-state message (replaces `np.random` placeholder). Commit: `feat: dashboard production mode with admin gate and real confidence data`.
19. `[x]` **Docker hardening.** `infra` | `must` — non-root `appuser` (UID 10000) in both images; python one-liner healthchecks (verified `healthy` in single-worker smoke test serving champion `v1789976193`); dropped `version:` key; env passthrough with dev defaults; kept single `internal: true` network with recorded rationale; `.dockerignore` added (images were ~10GB with venv/caches). Full production-topology run deferred to VPS per #29; retrainer never started on the small host. Commit: `fix: harden Docker images and compose`.
20. `[x]` **Capture-data retention policy.** `docs`/`infra` | `should` — `scripts/purge_captures.py` (`--days`, `--dry-run`) nulls `raw_request`/`headers_json` past retention (default 30d), keeps structured columns; policy documented in README (host cron recommended); purged rows still convert via tested degraded path. Commit: `feat: capture-data retention policy and purge script`.

### Tier 4 -- Hygiene (Phase D)

21. `[x]` **CI.** `infra` | `must` — `.github/workflows/ci.yml`: ruff check + format check + pytest on push/PR (Python 3.11 via uv). Verifies live on first push. Commit: `ci: add GitHub Actions quality gates`.
22. `[x]` **mypy.** `infra` | `should` — mypy 1.13.0 in dev deps (locked), `[tool.mypy]` config with overrides for stub-less third-party libs; 47 errors → 0, including real finds (`--random-state` now seeded through learners + folds, `None`-able review ids guarded, honest `| None` model types). CI runs mypy. Commits: `chore: add mypy config and get type checks green`, `ci: run mypy in quality gates`.
23. `[x]` **Model retention.** `infra` | `should` — `prune_models(keep_last_n=3)` keeps champion + N newest versioned pairs (spec §9: strictly opt-in via `scripts/prune_models.py`, `--dry-run` supported, champion files never touched). Commit: `feat: opt-in model retention pruning`.
24. `[x]` **SQLite backups.** `infra` | `should` — `scripts/backup_dbs.py` hot-copies the six known DBs via the backup API into timestamped sets (`--keep-last 7`, `--dry-run`); missing DBs skipped; cron documented in README. Commit: `feat: SQLite hot-backup script with pruning`.
25. `[x]` **Docs & dead-code accuracy.** `docs`/`app` | `should` — removed all-None feature maps (~130 lines), `reference_window`, `feature_stats` table, always-0 `decoy_indicator` (schema, logging, dashboard, tests); fixed double-save on promotion, `--random-state` propagation, validation-overwrite guard (+ dashboard message), UNSW `DtypeWarning` (`dtype=str`, verified); `requirements.txt` marked generated (pins match `pyproject.toml`); README + AGENTS.md run commands fixed. Net −105 lines. Commit: `chore: remove dead code and fix stale docs`.

### Tier 5 -- Release gates

26. `[x]` **Add Apache-2.0 LICENSE** (full text) at repo root. `docs` | `must` — `LICENSE` added (standard text, appendix boilerplate with project copyright).
27. `[x]` **Attribution / NOTICE.** `docs` | `must` — decision: `NOTICE` file added (dataset attributions + license pointer); README citations (L121-122) kept as the long form. Dataset files themselves are never redistributed (git-ignored `data/raw/`).
28. `[x]` **Tracking-docs fate + commit.** `infra` | `should` — decision: session plans under `docs/superpowers/` are working notes, excluded from the release via `.gitignore`; the curated record (`BACKLOG.md`, `docs/baseline-report.md`) stays tracked.
29. `[x]` **Final gates + flip public.** `infra` | `must` — final gates green (103 passed, ruff + mypy clean, tree clean, secrets scan clean); pushed to origin; visibility flipped PRIVATE → PUBLIC via `gh repo edit`. (VPS ops live in the Deferred checklist below.)

**Verified this pass (2026-09-22):** gitignore covers `data/`, `*.db*`, `*.npz`, `*.joblib`, `*.bin`, `.env`, `docs/superpowers/`; secrets scan clean; quality gates green (ruff + mypy clean, 103 tests passed). Model versioning works end-to-end.

### Monitor-only

- `--` **"Benign" class frozen from public datasets.** `docs` | monitor — by design; retraining only sees captured honeypot (attack-only) + public data, so drift detection is the only adaptive lever on benigns. Revisit if a benign-capture source is added (e.g. mirroring real traffic to a twin service).

### Deferred — post-release follow-ups (all work items done; these need time, hardware, or humans)

- `[ ]` **VPS deploy checklist.** `infra` | `must` (ops) — provision the VPS: **Hetzner CX33 (4 vCPU / 8 GB / 80 GB, ~€8.49/mo, Germany — verify pricing at hetzner.com/cloud; CX23 2vCPU/4GB ~€5.49 is the tight minimum)**. Rationale: retrainer capped at 2G, full-size retrain OOMs at 3.8G, images are ~10G each. Then: one-time `mirage-data` volume `chown` if migrating root-owned data; production `.env` (`MIRAGE_SECRET_KEY`/`MIRAGE_API_KEY`/`MIRAGE_ADMIN_KEY`, never dev defaults); full production-topology `compose up` with green healthchecks; enable the retrainer (never started on the small dev host); first backup + purge cron. Monitor the abuse-contact email in the first weeks (Hetzner enforcement is automated). Carried from #19/#29.
- `[ ]` **Legal review before going live.** `docs` | `must` (human) — spec §14: Nigerian Cybercrimes Act 2015 posture unresolved; consult someone knowledgeable before capturing real traffic. Post the `security.txt`/README disclosure notice at deploy time regardless.
- `[ ]` **Live-demo milestone.** `app` | `must` (time) — spec §19: 2–4 weeks of real honeypot traffic plus at least one genuine drift → review → retrain → promotion cycle. This is the only event that populates payload-keyword features (identically zero in all public flow data; the live champion scores real SQLi probes 0.003) and fills the dashboard drift/retraining panels with genuine history.
- `[ ]` **Re-run honesty studies on the new pipeline.** `app`/`docs` | `should` — `scripts/evaluate.py` temporal/cross/zero-features modes were measured against the pre-fix mapping; re-run against `v1789976193`'s pipeline and append to `docs/baseline-report.md`. Open since the 2026-09-21 retrain.
- `[ ]` **Full-size retrain on bigger hardware.** `app` | `could` — the live champion trained at 0.05/0.25 samples after two full-size (0.1/0.5) OOM kills on the 3.8 GiB host; a ≥8G machine could train the full 1.48M rows and test whether the 0.9708 F1 holds at full volume.
- `[ ]` **Distribution write-up.** `docs` | `must` (spec §19 milestone 3 — the actual finish line) — Dev.to/blog post + community share, linked from README and LinkedIn. Draft outline agreed 2026-09-22: hook on the fabrication twist ("0.92 AUC, learned nothing"), architecture brief, honesty-audit results (temporal 0.64, cross 0.18–0.49, ablation chain), the fix (F1 0.9708, old champion at 0.0024), the honest regression (SQLi probe 0.003), portable takeaways. Expand to full draft, publish, link.
- `[ ]` **`score_log` retention.** `app` | `should` — one row per `/api/score` call accumulates unbounded in `dashboard.db`; decide: extend the purge script, cap rows, or record explicit wont-fix (rows are tiny but the endpoint is internet-facing).

### Dropped / merged (v3.0 restructure)

- old #10 (integration test) -> superseded by #16 (end-to-end loop test; spec §16 exists).
- old #13 (weak real-request scoring) -> merged into #11 (shortcut audit).
- old #14 (UNSW DtypeWarning) -> merged into #25 (docs & dead-code accuracy).
- old #9 (AGENTS.md run command) -> merged into #25.
