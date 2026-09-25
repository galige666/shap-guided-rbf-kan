from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from shap_rbf_kan.model_search import _run_epochs, fit_final_model, make_fit_predict


def _data(seed: int = 13):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(28, 12))
    y = 1.4 * x[:, 1] - 0.6 * x[:, 7] + 0.05 * rng.normal(size=28)
    return x, y


MODEL_PARAMETERS = {
    "svr": {"C": 10.0, "gamma": 0.01, "epsilon": 0.03},
    "xgboost": {
        "max_depth": 2,
        "learning_rate": 0.05,
        "n_estimators": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-4,
        "reg_lambda": 1.0,
    },
    "mlp": {
        "hidden_layer_sizes": [8],
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "batch_size": 8,
    },
    "1d-cnn": {
        "channels": 4,
        "kernel_size": 3,
        "dense_features": 6,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
    },
    "b-spline-kan": {
        "hidden_features": [5],
        "num_basis": 3,
        "grid_half_width": 2.5,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
    },
}


@pytest.mark.parametrize("model", list(MODEL_PARAMETERS))
def test_every_search_model_returns_finite_fold_predictions(model):
    x, y = _data()
    fit_predict = make_fit_predict(
        model,
        max_epochs=3,
        min_epochs=1,
        patience=1,
        mlp_max_epochs=8,
        mlp_patience=2,
    )

    prediction = fit_predict(
        x[:22],
        y[:22],
        x[22:],
        MODEL_PARAMETERS[model],
        fold_seed=19,
    )

    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()


def test_scalers_are_fitted_only_on_supplied_training_samples():
    x, y = _data()
    fitted = fit_final_model(
        "svr",
        x[:20],
        y[:20],
        MODEL_PARAMETERS["svr"],
        seed=23,
    )

    np.testing.assert_allclose(fitted.x_scaler.mean_, x[:20].mean(axis=0))
    assert not np.allclose(fitted.x_scaler.mean_, x.mean(axis=0))


@pytest.mark.parametrize("model", ["mlp", "1d-cnn", "b-spline-kan"])
def test_neural_models_select_an_epoch_internally_then_refit(model):
    x, y = _data()
    fitted = fit_final_model(
        model,
        x,
        y,
        MODEL_PARAMETERS[model],
        seed=29,
        max_epochs=4,
        min_epochs=1,
        patience=1,
        mlp_max_epochs=8,
        mlp_patience=2,
    )

    assert fitted.selected_epoch is not None
    assert fitted.selected_epoch >= 1
    assert np.isfinite(fitted.predict(x[:3])).all()


def test_final_fit_is_reproducible_and_has_no_prediction_label_input():
    x, y = _data()
    first = fit_final_model(
        "b-spline-kan",
        x,
        y,
        MODEL_PARAMETERS["b-spline-kan"],
        seed=37,
        max_epochs=3,
        min_epochs=1,
        patience=1,
    )
    second = fit_final_model(
        "b-spline-kan",
        x,
        y,
        MODEL_PARAMETERS["b-spline-kan"],
        seed=37,
        max_epochs=3,
        min_epochs=1,
        patience=1,
    )

    np.testing.assert_allclose(first.predict(x[:5]), second.predict(x[:5]), atol=0, rtol=0)


def test_fixed_epoch_refit_can_reuse_the_selection_scheduler_horizon(monkeypatch):
    recorded_horizons = []
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    def recording_scheduler(optimizer, *args, **kwargs):
        recorded_horizons.append(kwargs["T_max"])
        return original_scheduler(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", recording_scheduler)
    model = torch.nn.Linear(3, 1)
    _run_epochs(
        model,
        np.ones((6, 3)),
        np.ones(6),
        epochs=2,
        scheduler_horizon=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=3,
        seed=7,
        device=torch.device("cpu"),
    )

    assert recorded_horizons == [30]


def _write_cli_tables(directory: Path, prediction_offset: float = 0.0):
    directory.mkdir(parents=True, exist_ok=True)
    x, y = _data()
    columns = [f"wavenumber_{index}" for index in range(x.shape[1])]
    calibration = pd.DataFrame(x[:22], columns=columns)
    calibration["target"] = y[:22]
    prediction = pd.DataFrame(x[22:], columns=columns)
    prediction["target"] = y[22:] + prediction_offset
    calibration_path = directory / "calibration.csv"
    prediction_path = directory / "prediction.csv"
    calibration.to_csv(calibration_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    return calibration_path, prediction_path


def test_bayesian_search_cli_writes_resumable_study_trials_and_summary(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    calibration_path, prediction_path = _write_cli_tables(tmp_path)
    output_directory = tmp_path / "search"
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_bayesian_search.py"),
        "--calibration-csv",
        str(calibration_path),
        "--prediction-csv",
        str(prediction_path),
        "--target",
        "target",
        "--model",
        "svr",
        "--trials",
        "2",
        "--folds",
        "3",
        "--output-dir",
        str(output_directory),
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary_path = output_directory / "fruit_svr_summary.json"
    trials_path = output_directory / "fruit_svr_trials.csv"
    database_path = output_directory / "optuna_studies.sqlite3"
    assert summary_path.exists()
    assert trials_path.exists()
    assert database_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trials = pd.read_csv(trials_path)
    assert summary["search"]["sampler"] == "TPESampler"
    assert summary["search"]["pruning"] == "disabled"
    assert summary["search"]["completed_trials"] == 2
    assert summary["search"]["target_trials"] == 2
    assert len(trials) == 2
    assert summary["prediction_metrics"] is not None

    resumed = subprocess.run(
        [*command, "--trials", "3", "--resume"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    resumed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert resumed_summary["search"]["completed_trials"] == 3


def test_prediction_labels_cannot_change_search_or_best_parameters(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    calibration_path, first_prediction = _write_cli_tables(tmp_path / "first")
    _, second_prediction = _write_cli_tables(tmp_path / "second", prediction_offset=10_000.0)

    summaries = []
    for name, prediction_path in (("first", first_prediction), ("second", second_prediction)):
        output_directory = tmp_path / f"output-{name}"
        completed = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "run_bayesian_search.py"),
                "--calibration-csv",
                str(calibration_path),
                "--prediction-csv",
                str(prediction_path),
                "--target",
                "target",
                "--model",
                "svr",
                "--trials",
                "2",
                "--folds",
                "3",
                "--seed",
                "101",
                "--output-dir",
                str(output_directory),
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        summaries.append(
            json.loads(
                (output_directory / "fruit_svr_summary.json").read_text(encoding="utf-8")
            )
        )

    assert summaries[0]["best_trial"] == summaries[1]["best_trial"]
    assert summaries[0]["prediction_metrics"] != summaries[1]["prediction_metrics"]


def test_cli_rejects_zero_trials_instead_of_running_the_default_budget(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    calibration_path, _ = _write_cli_tables(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_bayesian_search.py"),
            "--calibration-csv",
            str(calibration_path),
            "--target",
            "target",
            "--model",
            "svr",
            "--trials",
            "0",
            "--folds",
            "3",
            "--output-dir",
            str(tmp_path / "zero-trials"),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "positive" in completed.stderr.lower()
