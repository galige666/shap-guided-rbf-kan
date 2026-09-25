from __future__ import annotations

import numpy as np
import optuna
import pytest

from shap_rbf_kan.tuning import (
    SEARCH_BUDGETS,
    bayesian_search,
    suggest_hyperparameters,
)


EXPECTED_KEYS = {
    "svr": {"C", "gamma", "epsilon"},
    "xgboost": {
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    },
    "mlp": {
        "hidden_layer_sizes",
        "activation",
        "alpha",
        "learning_rate_init",
        "batch_size",
    },
    "1d-cnn": {
        "channels",
        "kernel_size",
        "dense_features",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
    },
    "b-spline-kan": {
        "hidden_features",
        "num_basis",
        "grid_half_width",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
    },
}


def test_declared_trial_budgets_match_the_revision_protocol():
    assert SEARCH_BUDGETS == {
        "svr": 48,
        "xgboost": 36,
        "mlp": 24,
        "1d-cnn": 24,
        "b-spline-kan": 40,
    }


def test_every_model_declares_a_runnable_optuna_space():
    for model, expected_keys in EXPECTED_KEYS.items():
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=17),
        )
        trial = study.ask()
        parameters = suggest_hyperparameters(trial, model, dataset="fruit")

        assert set(parameters) == expected_keys
        assert set(trial.params) == expected_keys
        assert "seed" not in parameters


def test_spaces_cover_wider_continuous_ranges_and_valid_architectures():
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=31),
    )
    svr_trial = study.ask()
    svr = suggest_hyperparameters(svr_trial, "svr")
    assert 0.1 <= svr["C"] <= 1_000.0
    assert 1e-5 <= svr["gamma"] <= 0.1
    assert 0.001 <= svr["epsilon"] <= 0.2

    kan_trial = study.ask()
    kan = suggest_hyperparameters(kan_trial, "b-spline-kan", dataset="oil")
    assert isinstance(kan["hidden_features"], list)
    assert kan["num_basis"] in {3, 5, 7, 9, 11}
    assert 2.0 <= kan["grid_half_width"] <= 5.0


def test_bayesian_search_runs_exact_trials_on_shared_five_fold_partitions():
    x = np.arange(60, dtype=float).reshape(30, 2)
    y = np.linspace(0.0, 1.0, 30)

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del params, fold_seed
        train_ids = set((x_train[:, 0] / 2).astype(int))
        valid_ids = (x_valid[:, 0] / 2).astype(int)
        assert train_ids.isdisjoint(set(valid_ids))
        return np.full(x_valid.shape[0], y_train.mean())

    study = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=3,
        folds=5,
        seed=41,
    )

    assert len(study.trials) == 3
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    for trial in study.trials:
        fold_rmse = trial.user_attrs["fold_rmse"]
        assert len(fold_rmse) == 5
        assert np.isclose(trial.value, np.mean(fold_rmse))
        assert np.isclose(trial.user_attrs["sd_rmse"], np.std(fold_rmse, ddof=0))


def test_tpe_parameter_sequence_is_reproducible_for_a_fixed_seed():
    x = np.arange(36, dtype=float).reshape(18, 2)
    y = np.linspace(-1.0, 1.0, 18)

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del x_train, params, fold_seed
        return np.full(x_valid.shape[0], y_train.mean())

    studies = [
        bayesian_search(
            x,
            y,
            model="svr",
            fit_predict=fit_predict,
            n_trials=4,
            folds=3,
            seed=73,
        )
        for _ in range(2)
    ]

    assert [trial.params for trial in studies[0].trials] == [
        trial.params for trial in studies[1].trials
    ]
    np.testing.assert_allclose(
        [trial.value for trial in studies[0].trials],
        [trial.value for trial in studies[1].trials],
        atol=0,
        rtol=0,
    )


def test_resume_rejects_changed_calibration_data_or_fold_protocol(tmp_path):
    x = np.arange(36, dtype=float).reshape(18, 2)
    y = np.linspace(-1.0, 1.0, 18)
    storage = f"sqlite:///{(tmp_path / 'study.sqlite3').as_posix()}"

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del x_train, params, fold_seed
        return np.full(x_valid.shape[0], y_train.mean())

    bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=1,
        folds=3,
        seed=83,
        storage=storage,
        study_name="resume-contract",
    )

    with pytest.raises(ValueError, match="does not match"):
        bayesian_search(
            x,
            y + 1.0,
            model="svr",
            fit_predict=fit_predict,
            n_trials=2,
            folds=3,
            seed=83,
            storage=storage,
            study_name="resume-contract",
            load_if_exists=True,
        )

    with pytest.raises(ValueError, match="does not match"):
        bayesian_search(
            x,
            y,
            model="svr",
            fit_predict=fit_predict,
            n_trials=2,
            folds=4,
            seed=83,
            storage=storage,
            study_name="resume-contract",
            load_if_exists=True,
        )


def test_resumed_tpe_matches_an_uninterrupted_parameter_sequence(tmp_path):
    x = np.arange(36, dtype=float).reshape(18, 2)
    y = np.linspace(-1.0, 1.0, 18)

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del x_train, params, fold_seed
        return np.full(x_valid.shape[0], y_train.mean())

    uninterrupted = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=12,
        folds=3,
        seed=97,
    )
    storage = f"sqlite:///{(tmp_path / 'resumed.sqlite3').as_posix()}"
    bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=6,
        folds=3,
        seed=97,
        storage=storage,
        study_name="resume-sequence",
    )
    resumed = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=12,
        folds=3,
        seed=97,
        storage=storage,
        study_name="resume-sequence",
        load_if_exists=True,
    )

    assert [trial.params for trial in resumed.trials] == [
        trial.params for trial in uninterrupted.trials
    ]
    assert resumed.user_attrs["n_startup_trials"] == 10


def test_resume_retains_failed_trials_and_reaches_completed_target(tmp_path):
    x = np.arange(36, dtype=float).reshape(18, 2)
    y = np.linspace(-1.0, 1.0, 18)
    storage = f"sqlite:///{(tmp_path / 'failed.sqlite3').as_posix()}"

    def fail_fit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("simulated worker failure")

    with pytest.raises(RuntimeError, match="simulated worker failure"):
        bayesian_search(
            x,
            y,
            model="svr",
            fit_predict=fail_fit,
            n_trials=2,
            folds=3,
            seed=101,
            storage=storage,
            study_name="failed-resume",
        )

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del x_train, params, fold_seed
        return np.full(x_valid.shape[0], y_train.mean())

    resumed = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=2,
        folds=3,
        seed=101,
        storage=storage,
        study_name="failed-resume",
        load_if_exists=True,
    )

    states = [trial.state for trial in resumed.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1
    assert states.count(optuna.trial.TrialState.COMPLETE) == 2


@pytest.mark.parametrize("recorded_parameters", ["none", "partial", "complete"])
def test_resume_recovers_a_stale_running_trial(tmp_path, recorded_parameters):
    x = np.arange(36, dtype=float).reshape(18, 2)
    y = np.linspace(-1.0, 1.0, 18)
    storage = f"sqlite:///{(tmp_path / 'running.sqlite3').as_posix()}"

    def fit_predict(x_train, y_train, x_valid, params, *, fold_seed):
        del x_train, params, fold_seed
        return np.full(x_valid.shape[0], y_train.mean())

    initial = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=1,
        folds=3,
        seed=103,
        storage=storage,
        study_name="running-resume",
    )
    interrupted = initial.ask()
    if recorded_parameters == "partial":
        interrupted.suggest_float("C", 0.1, 1_000.0, log=True)
    elif recorded_parameters == "complete":
        suggest_hyperparameters(interrupted, "svr")

    resumed = bayesian_search(
        x,
        y,
        model="svr",
        fit_predict=fit_predict,
        n_trials=2,
        folds=3,
        seed=103,
        storage=storage,
        study_name="running-resume",
        load_if_exists=True,
    )

    states = [trial.state for trial in resumed.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1
    assert states.count(optuna.trial.TrialState.COMPLETE) == 2
