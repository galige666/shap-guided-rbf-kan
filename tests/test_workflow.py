from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import subprocess
import sys

from shap_rbf_kan.workflow import fit_rbf_kan, run_shap_guided_workflow


def _synthetic_regression(seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(36, 6))
    y = 1.5 * x[:, 0] - 0.7 * x[:, 2] + 0.1 * rng.normal(size=36)
    return x, y


def _small_config():
    return {
        "hidden_features": [5],
        "num_basis": 4,
        "grid_half_width": 2.5,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "seed": 13,
    }


def test_fit_uses_internal_calibration_split_and_returns_finite_predictions():
    x, y = _synthetic_regression()

    fitted = fit_rbf_kan(
        x,
        y,
        _small_config(),
        max_epochs=12,
        min_epochs=3,
        patience=3,
    )
    prediction = fitted.predict(x[:4])

    assert prediction.shape == (4,)
    assert np.isfinite(prediction).all()
    assert set(fitted.train_indices).isdisjoint(fitted.validation_indices)
    assert sorted([*fitted.train_indices, *fitted.validation_indices]) == list(range(len(y)))
    assert 1 <= fitted.selected_epoch <= 12


def test_prediction_labels_do_not_change_training_or_checkpoint_selection():
    x, y = _synthetic_regression()
    x_prediction = x[:5] + 0.2
    supplied_shap = np.tile(np.arange(1.0, 7.0), (len(y), 1))

    first = run_shap_guided_workflow(
        x,
        y,
        config=_small_config(),
        top_k=3,
        shap_values=supplied_shap,
        prediction_x=x_prediction,
        prediction_y=np.zeros(5),
        max_epochs=8,
        min_epochs=2,
        patience=2,
    )
    second = run_shap_guided_workflow(
        x,
        y,
        config=_small_config(),
        top_k=3,
        shap_values=supplied_shap,
        prediction_x=x_prediction,
        prediction_y=np.full(5, 1_000_000.0),
        max_epochs=8,
        min_epochs=2,
        patience=2,
    )

    np.testing.assert_array_equal(first.selected_features, [5, 4, 3])
    np.testing.assert_array_equal(first.model_feature_indices, [3, 4, 5])
    np.testing.assert_allclose(first.compact_model.x_scaler.mean, x[:, [3, 4, 5]].mean(axis=0))
    np.testing.assert_array_equal(first.selected_features, second.selected_features)
    np.testing.assert_allclose(first.full_predictions, second.full_predictions, atol=0, rtol=0)
    np.testing.assert_allclose(first.compact_predictions, second.compact_predictions, atol=0, rtol=0)
    assert first.full_model.selected_epoch == second.full_model.selected_epoch
    assert first.compact_model.selected_epoch == second.compact_model.selected_epoch


def test_supplied_shap_values_must_match_calibration_shape():
    x, y = _synthetic_regression()

    with pytest.raises(ValueError, match="SHAP values"):
        run_shap_guided_workflow(
            x,
            y,
            config=_small_config(),
            top_k=3,
            shap_values=np.ones((len(y) - 1, x.shape[1])),
            max_epochs=2,
            min_epochs=1,
            patience=1,
        )


def test_reference_cli_exposes_required_inputs():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "run_reference_workflow.py"), "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--calibration-csv" in result.stdout
    assert "--target" in result.stdout
    assert "--top-k" in result.stdout


def test_reference_cli_rejects_nonfinite_targets(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    table = tmp_path / "bad-target.csv"
    pd.DataFrame(
        {
            "w1": np.arange(5, dtype=float),
            "w2": np.arange(5, dtype=float) + 1,
            "target": [1.0, 2.0, np.inf, 4.0, 5.0],
        }
    ).to_csv(table, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_reference_workflow.py"),
            "--calibration-csv",
            str(table),
            "--target",
            "target",
            "--top-k",
            "1",
            "--max-epochs",
            "2",
            "--min-epochs",
            "1",
            "--patience",
            "1",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "finite" in result.stderr.lower()
