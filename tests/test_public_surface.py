from __future__ import annotations

from pathlib import Path


def test_readme_describes_bayesian_spaces_budgets_and_validation_boundary():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    for phrase in [
        "rbf-kan",
        "shap-guided",
        "svr",
        "xgboost",
        "mlp",
        "1d-cnn",
        "b-spline kan",
        "calibration-only five-fold",
        "optuna",
        "tpesampler",
        "pruning",
        "run_bayesian_search.py",
    ]:
        assert phrase in lowered
    for budget in ["48", "36", "24", "40"]:
        assert budget in readme
    assert "c:\\users\\" not in lowered
    assert "reproduces the manuscript tables" not in lowered
    assert "deterministic sample" not in lowered
    assert "candidate_space" not in lowered
    assert "pip install -e ." in lowered
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "scripts" / "run_bayesian_search.py").is_file()
