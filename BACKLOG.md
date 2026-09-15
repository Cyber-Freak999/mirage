# BACKLOG: Mirage

**Version:** 2.1 | **Last Updated:** 2026-09-15
**Source:** product-spec.md, README.md, AGENTS.md, docs/superpowers/plans/2026-09-06-train-baseline-model.md

Status: `[x]` done, `[ ]` not done, `[~]` partial, `[c]` blocked, `--` removed/monitor-only.

> **Single source of truth** for all pending project work (training milestone, release prep, deferred items). The plan doc `docs/superpowers/plans/2026-09-06-train-baseline-model.md` stays the execution detail for Tier 0; its older inline Backlog section is superseded by this file.

---

## How to use this backlog

- **Tier** — priority band: `T0` active milestone, `T1` release-blocking/quality, `T2` deferred.
- **Component** -- which part it touches (`app`, `dashboard`, `infra`, `docs`, `tests`).
- **Priority** -- `must` / `should` / `could`.
- Items are ordered top-to-bottom; work proceeds in order, blocking dependencies noted inline.

---

## Master Priority Order (all pending, top-first)

### Tier 0 -- Finish the training milestone

1. `[x]` **CICIDS `inf` -> 0 (TDD fix).** `app` | `must` — coerce `±inf -> NaN -> 0` on each sampled chunk (`app/schema/datasets.py`); all 7 CICIDS files load at 0.1. Commit: `fix: coerce inf to zero when loading CICIDS2017` (`4850502`).
2. `[x]` **Re-measure combined load + split.** `infra` | `must` — peak 678 MiB (693,784 KB) at `--cicids-sample 0.1 --unsw-sample 0.5`; no swap used; fits host budget.
3. `[x]` **Run combined baseline.** `app` | `must` — trained 1,187,060 rows (combined 1,483,826 rows, 0.1 CICIDS / 0.5 UNSW, k-folds 5, val 0.2); wall clock 1:08:25; promoted as `v1789430774`.
4. `[x]` **Verify artifacts + API.** `app` | `must` — `champion.pkl|json` (`v1789430774`), `v1789430774.pkl|json`, `validation_set.npz`; `validation.db` rows `promoted=1` (`v1788810565` -> `v1789324008` -> `v1789430774`); `POST /api/score` HTTP 200 with new `model_version` (SQLi payload scored attack 0.76).
5. `[x]` **Record baseline report.** `docs` | `must` — `docs/baseline-report.md` (val F1 0.5637 / precision 0.4013 / recall 0.9465 / AUC 0.9240, row counts, class distribution, training time, top-8 features).
6. `[x]` **Supersede placeholder champion.** `app` | `must` — `v1788810565` archived; live `champion` is `v1789430774`; API serves the new version.

### Tier 1 -- Release prep + quality

7. `[ ]` **Add Apache-2.0 LICENSE** (full text) at repo root. `docs` | `must` — currently absent.
8. `[~]` **Attribution / NOTICE.** `docs` | `must` — CICIDS2017 + UNSW-NB15 citations and the UNSW academic-use restriction verified present (README L100-101); decide whether a NOTICE file is needed and add it if so.
9. `[ ]` **AGENTS.md run command is stale.** `docs` | `should` — `uv run python -m app` fails (no `app/__main__.py`); document `uv run flask --app "app:create_app()" run --port 5000`.
10. `[ ]` **Integration test (spec §16).** `tests` | `should` — **the §16 section is currently missing from `product-spec.md`**; align the spec first, then write the integration test.
11. `[ ]` **Tracking-docs fate + commit.** `infra` | `should` — decide whether `docs/superpowers/plans/*.md` remain repo artifacts; commit `BACKLOG.md` + `docs/` (or relocate) before release.
12. `[ ]` **Final gates + flip public.** `infra` | `must` — re-run `uv run ruff check .` + `uv run pytest tests/`, confirm `git status` clean, then flip repo visibility.

**Verified this pass (2026-09-13, not pending):** gitignore covers `data/`, `*.db*`, `*.npz`, `*.joblib`, `*.bin`, `.env`; secrets scan clean (no `.env`, no tracked key/secret patterns); quality gates green (ruff clean, 43 tests passed).

### Tier 2 -- Deferred / known issues

13. `[ ]` **Weak real-request scoring.** `app` | `could` — SQLi 0.49 / RCE 0.11 via API; the synthetic-HTTP mapping of flow data does not transfer to actual payloads. Investigate feature mapping or score calibration.
14. `[ ]` **UNSW load `DtypeWarning` (cols 1/3/47).** `app` | `could` — cosmetic load-time warnings.
15. `--` **"Benign" class frozen from public datasets.** `docs` | monitor — by design; retraining only sees captured honeypot + public data, so drift detection is the only adaptive lever on benigns.