from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json

import numpy as np
import optuna
from sklearn.model_selection import KFold


SEARCH_BUDGETS = {
    "svr": 48,
    "xgboost": 36,
    "mlp": 24,
    "1d-cnn": 24,
    "b-spline-kan": 40,
}

MODEL_NAMES = tuple(SEARCH_BUDGETS)
SEARCH_SPACE_VERSION = "2026-09-25-v1"

_MLP_ARCHITECTURES = ("64", "128", "64x32", "96x32", "128x64", "128x64x32")
_KAN_ARCHITECTURES = {
    "fruit": ("8", "12", "16", "24", "32", "16x8", "32x12", "48x16"),
    "oil": ("16", "24", "32", "48", "64", "96", "32x12", "64x16"),
}


def _normalise_model_name(model: str) -> str:
    key = model.strip().lower()
    if key not in SEARCH_BUDGETS:
        raise ValueError(f"unknown model '{model}'; choose from {', '.join(MODEL_NAMES)}")
    return key


def _decode_architecture(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(width) for width in value]
    return [int(width) for width in str(value).split("x")]


def decode_hyperparameters(
    model: str,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    """Convert Optuna-safe categorical strings into model-ready values."""

    key = _normalise_model_name(model)
    decoded = dict(parameters)
    architecture_key = {
        "mlp": "hidden_layer_sizes",
        "b-spline-kan": "hidden_features",
    }.get(key)
    if architecture_key and architecture_key in decoded:
        decoded[architecture_key] = _decode_architecture(decoded[architecture_key])
    return decoded


def suggest_hyperparameters(
    trial: optuna.Trial,
    model: str,
    *,
    dataset: str = "fruit",
) -> dict[str, object]:
    """Draw one configuration from the declared Optuna search space."""

    key = _normalise_model_name(model)
    dataset_key = dataset.strip().lower()
    if key == "svr":
        parameters: dict[str, object] = {
            "C": trial.suggest_float("C", 0.1, 1_000.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 0.1, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.001, 0.2, log=True),
        }
    elif key == "xgboost":
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1_200, step=50),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.05
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    elif key == "mlp":
        parameters = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", _MLP_ARCHITECTURES
            ),
            "activation": trial.suggest_categorical("activation", ("relu", "tanh")),
            "alpha": trial.suggest_float("alpha", 1e-6, 0.1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-5, 0.01, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", (32, 64, 128)),
        }
    elif key == "1d-cnn":
        parameters = {
            "channels": trial.suggest_categorical("channels", (8, 12, 16, 24, 32, 48)),
            "kernel_size": trial.suggest_categorical(
                "kernel_size", (3, 5, 7, 9, 13, 15)
            ),
            "dense_features": trial.suggest_categorical(
                "dense_features", (16, 32, 48, 64, 96, 128)
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.005, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.01, log=True),
            "batch_size": trial.suggest_categorical("batch_size", (32, 64, 128)),
        }
    else:
        if dataset_key not in _KAN_ARCHITECTURES:
            raise ValueError("dataset must be 'fruit' or 'oil' for B-spline KAN")
        parameters = {
            "hidden_features": trial.suggest_categorical(
                "hidden_features", _KAN_ARCHITECTURES[dataset_key]
            ),
            "num_basis": trial.suggest_categorical("num_basis", (3, 5, 7, 9, 11)),
            "grid_half_width": trial.suggest_float(
                "grid_half_width", 2.0, 5.0, step=0.5
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.025),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.005, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", (32, 64, 128)),
        }
    return decode_hyperparameters(key, parameters)


FitPredict = Callable[..., np.ndarray]


def _calibration_fingerprint(features: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, targets):
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(json.dumps(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _make_sampler(seed: int, startup_trials: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=startup_trials,
    )


def _replay_sampler(
    historical_study: optuna.Study,
    *,
    model: str,
    dataset: str,
    seed: int,
    startup_trials: int,
) -> optuna.samplers.TPESampler:
    """Restore deterministic sampler state from completed sequential trials."""

    sampler = _make_sampler(seed, startup_trials)
    shadow = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )
    for historical in historical_study.trials:
        if historical.state not in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
        }:
            raise ValueError(
                f"stored study contains unsupported {historical.state.name} trial state"
            )
        replayed = shadow.ask()
        if historical.state == optuna.trial.TrialState.COMPLETE:
            suggest_hyperparameters(replayed, model, dataset=dataset)
        else:
            # A killed process can leave zero, some, or all suggestions stored.
            # Replaying only those recorded distributions advances the pinned
            # Optuna sampler by exactly the work performed before interruption.
            for name, distribution in historical.distributions.items():
                replayed._suggest(name, distribution)  # noqa: SLF001
        if replayed.params != historical.params:
            raise ValueError(
                "stored study sampling history does not match the declared protocol"
            )
        if historical.state == optuna.trial.TrialState.COMPLETE:
            if historical.value is None:
                raise ValueError("stored completed trial has no objective value")
            shadow.tell(replayed, float(historical.value))
        else:
            shadow.tell(replayed, state=optuna.trial.TrialState.FAIL)

    # Optuna 5.0 binds this cache to the shadow study ID. Recreate only the
    # cache before attaching the sampler to the persistent study; both random
    # generators keep the state advanced by the replay above.
    sampler._search_space = optuna.search_space.IntersectionSearchSpace(  # noqa: SLF001
        include_pruned=True
    )
    return sampler


def select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Select minimum mean RMSE, using fold RMSE SD as a deterministic tie-breaker."""

    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete:
        raise ValueError("the study contains no completed trials")
    return min(
        complete,
        key=lambda trial: (
            float(trial.value),
            float(trial.user_attrs.get("sd_rmse", float("inf"))),
            trial.number,
        ),
    )


def bayesian_search(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    fit_predict: FitPredict,
    dataset: str = "fruit",
    n_trials: int | None = None,
    folds: int = 5,
    seed: int = 20260726,
    storage: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = False,
    protocol_metadata: Mapping[str, object] | None = None,
) -> optuna.Study:
    """Minimize mean calibration-only K-fold RMSE with a seeded TPE sampler.

    When a stored study is resumed, ``n_trials`` is interpreted as the required
    total number of completed trials rather than the number of additional trials.
    """

    key = _normalise_model_name(model)
    dataset_key = dataset.strip().lower()
    features = np.asarray(x, dtype=float)
    targets = np.asarray(y, dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[0] != targets.size:
        raise ValueError("x and y must contain the same number of paired samples")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("x and y must contain only finite values")
    if not 2 <= folds <= features.shape[0]:
        raise ValueError("folds must be between 2 and the number of samples")
    budget = SEARCH_BUDGETS[key] if n_trials is None else int(n_trials)
    if budget < 1:
        raise ValueError("n_trials must be positive")

    metadata = dict(protocol_metadata or {})
    try:
        metadata_json = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as error:
        raise ValueError("protocol_metadata must contain JSON-serializable values") from error
    startup_trials = min(10, SEARCH_BUDGETS[key])
    immutable_protocol = {
        "model": key,
        "dataset": dataset_key,
        "folds": folds,
        "sampler": "TPESampler",
        "sampler_seed": seed,
        "n_startup_trials": startup_trials,
        "optuna_version": optuna.__version__,
        "search_space_version": SEARCH_SPACE_VERSION,
        "calibration_fingerprint": _calibration_fingerprint(features, targets),
        "protocol_metadata_json": metadata_json,
    }

    existing_study = None
    if load_if_exists and storage is not None and study_name is not None:
        try:
            existing_study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            existing_study = None

    if existing_study is not None:
        for name, expected in immutable_protocol.items():
            if existing_study.user_attrs.get(name) != expected:
                raise ValueError(
                    f"stored study protocol does not match the current {name}"
                )
        for historical in existing_study.trials:
            if historical.state == optuna.trial.TrialState.RUNNING:
                existing_study.tell(
                    historical.number,
                    state=optuna.trial.TrialState.FAIL,
                    skip_if_finished=True,
                )
        sampler = _replay_sampler(
            existing_study,
            model=key,
            dataset=dataset_key,
            seed=seed,
            startup_trials=startup_trials,
        )
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )
    else:
        sampler = _make_sampler(seed, startup_trials)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            storage=storage,
            study_name=study_name,
            load_if_exists=False,
        )
        for name, value in immutable_protocol.items():
            study.set_user_attr(name, value)
        study.set_user_attr("protocol_metadata", metadata)

    study.set_user_attr("target_completed_trials", budget)
    study.set_user_attr("stopping_rule", "complete the declared trial budget")
    study.set_user_attr("pruning", "disabled")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(features))

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_hyperparameters(trial, key, dataset=dataset_key)
        fold_rmse: list[float] = []
        for fold_index, (train_index, validation_index) in enumerate(splits):
            prediction = np.asarray(
                fit_predict(
                    features[train_index],
                    targets[train_index],
                    features[validation_index],
                    parameters,
                    fold_seed=seed + fold_index + 1,
                ),
                dtype=float,
            ).reshape(-1)
            if prediction.size != validation_index.size or not np.isfinite(prediction).all():
                raise ValueError("fit_predict must return one finite prediction per validation sample")
            rmse = float(
                np.sqrt(np.mean((targets[validation_index] - prediction) ** 2))
            )
            fold_rmse.append(rmse)
        mean_rmse = float(np.mean(fold_rmse))
        trial.set_user_attr("fold_rmse", fold_rmse)
        trial.set_user_attr("mean_rmse", mean_rmse)
        trial.set_user_attr("sd_rmse", float(np.std(fold_rmse, ddof=0)))
        trial.set_user_attr("fold_seeds", [seed + index + 1 for index in range(folds)])
        return mean_rmse

    completed = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    if completed > budget:
        raise ValueError(
            f"stored study already has {completed} completed trials, exceeding budget {budget}"
        )
    remaining = budget - completed
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    return study
