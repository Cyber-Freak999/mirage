"""Tests for retrainer entrypoint defaults and OOM-safe sample caps."""

from app.retrain.scheduler import RetrainingConfig


def test_retrain_config_defaults_fit_container() -> None:
    """Default sample caps stay within the 2G retrainer container budget."""
    config = RetrainingConfig()

    assert config.cicids_sample <= 0.05
    assert config.unsw_sample <= 0.25


def test_entry_config_uses_dataset_dirs_when_present(tmp_path, monkeypatch) -> None:
    """scheduler_entry points at data/raw dirs instead of honeypot-only None."""
    import app.retrain.scheduler_entry as entry_mod

    cicids = tmp_path / "raw" / "cicids2017"
    unsw = tmp_path / "raw" / "unsw-nb15"
    cicids.mkdir(parents=True)
    unsw.mkdir(parents=True)
    monkeypatch.setattr(entry_mod, "DATA_ROOT", tmp_path)

    config = entry_mod.build_config()

    assert config.cicids_dir == cicids
    assert config.unsw_dir == unsw


def test_process_review_passes_sample_caps(tmp_path, monkeypatch) -> None:
    """process_review forwards the config sample caps to run_full_retrain."""
    import app.api as api_mod
    import app.honeypot as honeypot_mod
    import app.retrain.drift as drift_mod
    import app.retrain.review_gate as review_mod
    from app.retrain.champion_challenger import ValidationResult
    from app.retrain.scheduler import RetrainingScheduler

    monkeypatch.setattr(drift_mod, "DRIFT_DB", tmp_path / "drift.db")
    monkeypatch.setattr(review_mod, "REVIEW_DB", tmp_path / "review.db")
    monkeypatch.setattr(honeypot_mod, "DB_PATH", tmp_path / "honeypot.db")
    honeypot_mod.init_db()

    calls: dict = {}

    def fake_retrain(X_new, y_new, **kwargs):
        calls["kwargs"] = kwargs
        return None, ValidationResult(
            timestamp=0.0,
            challenger_version="v-test",
            champion_version="v-old",
            challenger_metrics={},
            champion_metrics={},
            promoted=False,
            reason="test",
        )

    monkeypatch.setattr("app.retrain.scheduler.run_full_retrain", fake_retrain)
    monkeypatch.setattr(api_mod, "reload_model", lambda: None)

    config = RetrainingConfig(
        cicids_sample=0.05,
        unsw_sample=0.25,
        cicids_dir=tmp_path / "cicids",
        unsw_dir=tmp_path / "unsw",
    )
    scheduler = RetrainingScheduler(config)
    gate = scheduler.review_gate
    review = gate.create_review(drift_result_id=1, max_psi=0.5, psi_scores={}, sample_size=0, sample_data={})
    gate.approve(review.id, "tester")

    scheduler.process_review(review.id, approved=True, reviewer="tester")

    assert calls["kwargs"]["cicids_sample"] == 0.05
    assert calls["kwargs"]["unsw_sample"] == 0.25
    assert calls["kwargs"]["cicids_dir"] == tmp_path / "cicids"
    assert calls["kwargs"]["unsw_dir"] == tmp_path / "unsw"
