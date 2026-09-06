# Mirage — Adaptive IDS Project Specification

**Type:** Portfolio project (originally scoped as a three-month academic project)
**Purpose:** Mirage is a honeypot-driven adaptive intrusion detection system. A high-interaction honeypot lures and captures real attack traffic, which — combined with public benchmark datasets — trains a stacking ensemble to detect intrusions. The system monitors its own performance and retrains itself when detection starts to drift, with a validation gate protecting against a bad retrain reaching production.

---

## 1. Architecture Overview

**Data / training pipeline:**
Honeypot entry points → high-interaction backend → logging & labeling → feature extraction (unified schema) → merged with public datasets → stacking ensemble

**Live / adaptive loop:**
Live scoring → drift monitor → review gate → full retrain → validation gate → deploy & dashboard → (loops back to live scoring)

---

## 2. Honeypot Design

- **Interaction level:** High-interaction — realistic simulated depth, not genuinely exploitable. Chosen over low-interaction now that the project isn't time-boxed by an academic deadline, and over *genuinely* vulnerable real services because the risk (pivot to real infrastructure, abuse) outweighs the marginal realism gain for an internet-facing portfolio demo.
- **Simulated backend, entry-point-specific:**
  - SQLi-style entry points → fake queryable database (decoy tables, logs every query attempt)
  - RCE/upload-style entry points → fake filesystem + limited fake shell (responds to common commands like `ls`, `cat`, `whoami`, `wget` with scripted plausible output; planted decoy files)
- **Phase 1 endpoint scope:** Two-and-two split
  - 2 SQLi-style endpoints (e.g., login form, search/filter) → fake DB
  - 2 RCE/upload-style endpoints (e.g., file upload, admin diagnostics panel) → fake filesystem/shell
  - Both attack families are HTTP-request-shaped, so they're covered by the existing 25-feature schema without any rework.
- **Phase 2 candidate (if scope expands later):** A schema-compatible family like XSS or path traversal — extends the existing schema rather than requiring a redesign.
- **Future "v2," not phase 2:** SSRF or deserialization. These would require a new *outbound-behavior* feature block (destination IP/port, internal-IP detection, redirect count, response latency, response fingerprinting), since their signal lives in what the server does *after* the request, not in the request itself. Also would need separate handling for public-dataset mapping (CICIDS2017/UNSW-NB15 have little/no SSRF coverage, so training data would be honeypot-only) and per-class retraining-threshold logic due to sparse data volume relative to SQLi/RCE.

---

## 3. Labeling

- All honeypot traffic is treated as **positive-class (attack) by default** — nothing reaches an unlisted, unauthenticated fake service without malicious or reconnaissance intent, so the attack/benign question is already answered by the traffic reaching the honeypot at all.
- A **rule-based signature matcher** assigns the attack **subtype** only (SQLi vs. RCE/upload), never the attack/benign decision itself.
- **Scanner/researcher noise:** Known scanner/researcher traffic (Shodan, Censys, academic crawlers) is tagged using their published IP ranges — kept visible in logs/dashboard for honest total-traffic context, but **excluded from the attack-subtype classifier's training data**, since automated background scanning is a different signal than targeted attack behavior.

---

## 4. Feature Schema

- **25-feature unified HTTP-layer schema**, shared across honeypot logs and both public datasets (CICIDS2017, UNSW-NB15), so all three sources train the model together in the same feature space.
- Covers request-level characteristics (payload length, special-character density, parameter entropy, request method/path patterns, etc.) — features that generalize across SQLi and RCE/upload traffic because both ultimately present as anomalous HTTP requests.

---

## 5. Benign-Class Source (known limitation)

- All benign/normal training examples come **entirely from the public datasets** — honeypot traffic is 100% positive-class by design, so it never contributes fresh benign data.
- **Documented limitation:** the model's notion of "benign" is frozen to when CICIDS2017/UNSW-NB15 were collected; only the attack side of the system actually adapts over time.
- **Mitigation:** an empirical validation step evaluating model performance split by data source, to check for source-based shortcut learning (e.g., the model learning "public-dataset-shaped = benign, honeypot-shaped = attack" as a shortcut rather than true attack signatures).
- Explicitly decided **against** building a synthetic benign-traffic generator — the build cost isn't proportionate to the benefit at this scale.

---

## 6. Class / Source Balance

- **Class weighting** (`class_weight` / `scale_pos_weight`) as the primary approach — avoids the synthetic-data realism risk of SMOTE and the data loss of undersampling.
- **Fixed sampling ratio cap per retrain** — prevents growing honeypot volume from silently swamping the public dataset's contribution (or vice versa) as retrains accumulate over months.

---

## 7. Detection Model — Stacking Ensemble

- **Base learners:** Random Forest + XGBoost (different error patterns — RF is robust to noise via independent trees, XGBoost catches subtler sequential-correction patterns).
- **Meta-learner:** Logistic Regression — learns when to trust each base learner more, based on their disagreement patterns; more expressive than fixed voting/averaging.
- **Why stacking over a single tuned model:** the complexity is justified specifically because this is a *portfolio* project — demonstrating rigorous ML engineering (proper k-fold stacking, leakage avoidance) is part of the point, not just raw predictive performance.
- **Leakage prevention:** k-fold cross-validated stacking — base learners trained on k-1 folds, predict on the held-out fold (repeated across all folds), meta-learner trains on out-of-fold predictions only; base learners retrained on full data for deployment.
- **k is a tunable hyperparameter** (test empirically, e.g., 3/5/10) rather than fixed at 5 — evaluated once real data volume is available.
- **Known limitation — adversarial evasion:** not actively defended against (no adversarial training). Documented explicitly, plus lightweight manual evasion testing once the model is trained (hand-crafted obfuscated SQLi/RCE payloads) to produce concrete findings for the write-up rather than a bare disclaimer.

---

## 8. Retraining Loop (Adaptive Mechanism)

- **Trigger:** Feature distribution drift — label-free (PSI or KL divergence between incoming and training feature distributions). Chosen over error rate against the signature matcher's own labels, which would be circular (the model was trained on labels derived from that same matcher, so both would share blind spots and agree on being wrong together).
- **Threshold:** PSI ≥ 0.25 (standard "significant drift" convention), computed on the **top 5-8 features by importance** from the trained ensemble — not all 25 — to avoid trigger fatigue.
- **Cadence:** Hybrid — runs once a **200-sample minimum or a 24-hour cap** is reached, whichever comes first. Both values are tunable once real traffic volume is observed (no prior expectation of volume exists yet).
- **Review gate before retraining:** When drift fires, the triggering batch is surfaced via an **admin-dashboard review queue** plus a lightweight notification (email/webhook). A human spot-checks a sample to confirm it's a real pattern, not noise — this is a sanity check on the *data*, not the model.
  - If not reviewed within **72 hours**, retraining **auto-proceeds anyway** — the manual review isn't the real safety net; the validation gate downstream is.
- **Retraining scope:** Full retrain from scratch on all accumulated data (honeypot + public datasets) each time — not incremental/warm-start. RF doesn't support clean incremental updates anyway, and full retraining is cheap at this data scale.
- **Deployment gate:** Champion/challenger validation — the retrained model is evaluated on a **fixed held-out validation set** and only promoted to production if it meets or beats the current live model's precision/recall/F1. No immediate swap, ever.
- **Known gap:** the 72-hour review timeout means a deliberately poisoned batch (data poisoning attempt) could reach the retraining step unreviewed — but it still has to beat the current model on held-out validation data to actually get deployed.
- **Known gap:** the drift trigger only watches the top 5-8 tracked features — drift in an untracked feature, or degradation visible only in prediction confidence, wouldn't fire the trigger.

---

## 9. Model Versioning & Rollback

- **Simple archive approach:** every promoted model is saved with a timestamp/version ID before overwriting the "current" live pointer.
- **Manual rollback:** if a promoted model turns out to be problematic in production, roll back by pointing serving code at a prior saved version.
- Explicitly decided **against** a full automated MLOps registry (e.g., MLflow) — over-engineering relative to what a solo portfolio project needs.

---

## 10. Serving

- **Real-time, per-request scoring** via the Flask API — every honeypot request is scored immediately against the live model. Chosen over batch/periodic scoring specifically because real-time detection is core to the "adaptive IDS" demo narrative. Inference is cheap; only training is expensive, so there's no performance reason to batch.
- **Framework:** Flask (not FastAPI) for both the honeypot's fake endpoints and the prediction-serving API — lightweight, unopinionated, keeps everything in the same Python/ML ecosystem, and is the most widely recognized framework for a reviewer skimming the code.

---

## 11. Dashboard

- **Built with Plotly Dash.**
- **Full scope — five panels:**
  1. Live attack feed (recent hits, entry point, detected subtype, model confidence)
  2. Model confidence / prediction trend over time
  3. Feature drift (PSI) visualization for the top 5-8 tracked features
  4. Retraining / model version history (each retrain attempt, pass/fail on the validation gate, performance deltas)
  5. Attack-type breakdown (SQLi vs. RCE/upload volume over time)
- **Build sequencing:** live feed + confidence trend first (populated from day one); drift + retraining panels added once real retraining history actually exists.
- **Access control:** Read-only public view for charts/feed. Admin-only auth gate for raw honeypot logs (e.g., attacker IPs) and any manual system controls (e.g., manually triggering a retrain).

---

## 12. Isolation Architecture (non-negotiable)

All five controls apply:
1. **Per-service Docker containers** — no shared filesystem or network namespace between services or the host.
2. **No outbound internet access from honeypot containers** — the single most important control.
3. **CPU/memory resource limits per container** — prevents a single attacker or automated scanner flood from exhausting host resources.
4. **No credential/secret reuse** — any planted decoy credentials, API keys, or config files must be entirely fabricated, never real or old rotated values.
5. **Narrow logging exception** — the only permitted egress from a honeypot container is the log-write path to the SQLite store.

---

## 13. Abuse Prevention

- **Per-IP rate limiting** at the entry point — protects container resource limits.
- **Separate per-source sampling cap in the training pipeline** — protects data quality even for traffic that stays within the rate limit.
- **All traffic is still logged regardless of rate limiting** — the cap affects what enters model training, not what gets recorded.

---

## 14. Legal & Deployment Posture

- **Deploy publicly**, but post a clear, visible notice (e.g., a `security.txt` file or README note) stating this is a defensive research honeypot, disclosing log retention practices, and stating captured data is used for security research only. This is standard practice among established honeypot projects (e.g., Cowrie).
- **Open item:** this does not resolve legal ambiguity under Nigerian cyber law (Cybercrimes Act 2015, as amended) specifically. A real conversation with someone knowledgeable in that area is recommended before public launch — this spec does not constitute legal certainty.

---

## 15. Dataset Licensing

- **CICIDS2017:** Requires citing the dataset paper — Sharafaldin, Habibi Lashkari, and Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
- **UNSW-NB15:** Free use granted for academic research purposes; commercial use is restricted or requires author agreement (terms vary slightly by source page). Requires citing Moustafa and Slay's original papers.
- **Action:** Cite both properly in the README (required for CICIDS2017, best practice and license compliance for UNSW-NB15), plus a one-line note acknowledging UNSW-NB15's non-commercial terms.

---

## 16. Testing Strategy

- **Unit tests:** the 25-feature schema mapping — confirm honeypot logs and both public datasets all produce features in matching shape/order/scale.
- **Integration test:** one end-to-end test — a synthetic request through a honeypot endpoint should end up correctly logged, labeled, feature-extracted, and scored.
- **Validation-check scripts (not classic pass/fail tests):** drift/PSI math (confirm it fires and doesn't fire against known synthetic distribution shifts); champion/challenger comparison logic (confirm correct promote/reject decisions against known performance sets).
- **Explicitly out of scope:** formal testing of ensemble predictive performance (that's evaluation, not testing) and Dash dashboard UI rendering (manual QA instead).

---

## 17. Hosting

- **Cheap VPS** (e.g., DigitalOcean, Hetzner, Linode — roughly $5-12/month). Chosen over a cloud free tier (networking/AUP restrictions around honeypot-like services) and a home server (exposes personal home IP, reliability risk).
- The isolation architecture (Docker network policies, no outbound access, resource limits) requires full networking control that a VPS provides cleanly.

---

## 18. Tech Stack Summary

| Layer | Choice |
|---|---|
| Honeypot + API framework | Flask |
| Storage | SQLite (WAL mode recommended for concurrent access) |
| Base models | Random Forest, XGBoost |
| Meta-learner | Logistic Regression |
| Scheduling | APScheduler (persistent job store required) |
| Dashboard | Plotly Dash |
| Containerization | Docker (isolated, no outbound internet) |
| Public datasets | CICIDS2017, UNSW-NB15 |
| Hosting | Small VPS (DigitalOcean/Hetzner/Linode) |

---

## 19. Success Criteria ("Done")

1. **Live-demo milestone:** honeypot has run continuously for 2-4 weeks and accumulated at least one real retraining event, so the dashboard shows genuine adaptive behavior.
2. **Documentation milestone:** polished README with architecture diagrams, decision rationale, limitations section (adversarial evasion, benign-class staleness, drift-trigger blind spots), and proper dataset citations.
3. **Distribution milestone (the actual finish line):** published somewhere visible — a blog/Dev.to write-up, a relevant community post, or linked prominently from GitHub/LinkedIn.
- A specific performance-metric threshold (e.g., F1 target) is tracked internally but does **not** gate completion — chasing it indefinitely on a small dataset has diminishing returns for a portfolio project.

---

## 20. Recommended Reading Before Building

**Load-bearing (would block or break the system if skipped):**
1. Docker container security & network isolation — Docker security docs; NIST SP 800-190
2. Cowrie honeypot source (GitHub) — closest real-world reference for realistic simulated depth
3. APScheduler persistent job stores — scheduled jobs are lost on restart by default
4. SQLite WAL mode — needed once multiple services read/write the store concurrently

**Core technical background:**
5. Honeypot design theory — The Honeynet Project (honeynet.org)
6. Ensemble learning / stacking — *An Introduction to Statistical Learning* (free PDF); scikit-learn `StackingClassifier` docs
7. Concept drift & distribution monitoring — Evidently AI (evidentlyai.com) for PSI/KL divergence
8. MLOps deployment patterns — Google's *Rules of ML*

**Build-time, learnable in parallel:**
9. Flask application factories/blueprints; Gunicorn/WSGI basics
10. Plotly Dash callbacks — official Dash tutorial
11. Reverse proxy + TLS — nginx + Let's Encrypt
12. Basic threat modeling — OWASP cheat sheet (e.g., STRIDE), applied to the honeypot's containment boundary
13. Legal/privacy norms around logging attacker IPs
14. Strong technical README writing — review a few well-regarded ML/security project READMEs

---

## 21. Suggested Build Order

1. Honeypot (minimal version, logging to SQLite) — everything downstream depends on real logs existing
2. Unified 25-feature schema implementation (honeypot logs + public datasets → same shape)
3. Baseline model (single Random Forest) to validate the pipeline end-to-end
4. Full stacking ensemble (add XGBoost + Logistic Regression meta-learner, k-fold cross-validated)
5. Retraining loop (drift monitor, review gate, champion/challenger validation)
6. Dashboard (live feed + confidence trend first, then drift + retraining panels)

---

## 22. Open / Deferred Items

- Exact best value of *k* for stacking — to be tuned empirically once real data exists
- Exact PSI threshold and drift-check cadence values — defaults set (0.25, 200-sample/24-hour hybrid), both explicitly tunable
- Phase 2 (schema-compatible attack family) and v2 (SSRF/deserialization) — deferred, not in current build scope
- Legal review of the honeypot's deployment under Nigerian cyber law — not yet obtained
