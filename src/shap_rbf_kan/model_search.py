from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
from torch import nn

from .models import BSplineKANRegressor, SpectralCNN
from .tuning import MODEL_NAMES, decode_hyperparameters


class _TorchMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        activation_type: type[nn.Module]
        if activation == "relu":
            activation_type = nn.ReLU
        elif activation == "tanh":
            activation_type = nn.Tanh
        else:
            raise ValueError("MLP activation must be 'relu' or 'tanh'")
        widths = [in_features, *(int(width) for width in hidden_features), 1]
        blocks: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(widths[:-1], widths[1:])):
            blocks.append(nn.Linear(source, target))
            if index < len(widths) - 2:
                blocks.append(activation_type())
        self.network = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class FittedSearchModel:
    """A fitted estimator with fold-local input and target standardization."""

    model_name: str
    estimator: Any
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    selected_epoch: int | None = None
    device: str = "cpu"

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        transformed = self.x_scaler.transform(features)
        if isinstance(self.estimator, nn.Module):
            self.estimator.eval()
            tensor = torch.as_tensor(
                transformed,
                dtype=torch.float32,
                device=torch.device(self.device),
            )
            with torch.no_grad():
                scaled = self.estimator(tensor).detach().cpu().numpy().reshape(-1, 1)
        else:
            scaled = np.asarray(self.estimator.predict(transformed), dtype=float).reshape(-1, 1)
        return self.y_scaler.inverse_transform(scaled).reshape(-1)


def _normalise_model_name(model: str) -> str:
    key = model.strip().lower()
    if key not in MODEL_NAMES:
        raise ValueError(f"unknown model '{model}'; choose from {', '.join(MODEL_NAMES)}")
    return key


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_torch_model(
    model: str,
    feature_count: int,
    parameters: Mapping[str, object],
) -> nn.Module:
    if model == "mlp":
        return _TorchMLP(
            feature_count,
            parameters["hidden_layer_sizes"],
            str(parameters["activation"]),
        )
    if model == "1d-cnn":
        return SpectralCNN(
            channels=int(parameters["channels"]),
            kernel_size=int(parameters["kernel_size"]),
            dense_features=int(parameters["dense_features"]),
            dropout=float(parameters["dropout"]),
        )
    if model == "b-spline-kan":
        half_width = float(parameters["grid_half_width"])
        return BSplineKANRegressor(
            feature_count,
            hidden_features=parameters["hidden_features"],
            num_basis=int(parameters["num_basis"]),
            grid_min=-half_width,
            grid_max=half_width,
            dropout=float(parameters["dropout"]),
        )
    raise ValueError(f"{model} is not a neural model")


def _training_schedule(
    model: str,
    *,
    max_epochs: int | None,
    min_epochs: int | None,
    patience: int | None,
    mlp_max_epochs: int | None,
    mlp_patience: int | None,
) -> tuple[int, int, int, float]:
    defaults = {
        "mlp": (2_000, 80, 80, 1e-5),
        "1d-cnn": (900, 100, 100, 1e-7),
        "b-spline-kan": (700, 80, 80, 1e-7),
    }
    default_max, default_min, default_patience, min_delta = defaults[model]
    resolved_max = (
        mlp_max_epochs
        if model == "mlp" and mlp_max_epochs is not None
        else max_epochs if max_epochs is not None else default_max
    )
    resolved_min = min_epochs if min_epochs is not None else min(default_min, resolved_max)
    resolved_patience = (
        mlp_patience
        if model == "mlp" and mlp_patience is not None
        else patience if patience is not None else default_patience
    )
    if resolved_max < 1 or not 1 <= resolved_min <= resolved_max or resolved_patience < 1:
        raise ValueError("neural epoch and patience settings are inconsistent")
    return resolved_max, resolved_min, resolved_patience, min_delta


def training_protocol(
    model: str,
    *,
    max_epochs: int | None = None,
    min_epochs: int | None = None,
    patience: int | None = None,
    mlp_max_epochs: int | None = None,
    mlp_patience: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Return the resolved model-fitting settings used in protocol fingerprints."""

    key = _normalise_model_name(model)
    common: dict[str, object] = {
        "implementation_version": "2026-09-25-v1",
        "feature_scaling": "fold-training StandardScaler",
        "target_scaling": "fold-training StandardScaler",
    }
    if key == "svr":
        return {**common, "solver": "sklearn SVR with RBF kernel"}
    if key == "xgboost":
        return {
            **common,
            "solver": "XGBRegressor",
            "tree_method": "hist",
            "n_jobs": 1,
        }
    resolved_max, resolved_min, resolved_patience, min_delta = _training_schedule(
        key,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
        mlp_max_epochs=mlp_max_epochs,
        mlp_patience=mlp_patience,
    )
    return {
        **common,
        "optimizer": "AdamW",
        "loss": "MSE",
        "scheduler": "CosineAnnealingLR",
        "scheduler_horizon": resolved_max,
        "max_epochs": resolved_max,
        "earliest_stopping_epoch": resolved_min,
        "patience": resolved_patience,
        "min_delta": min_delta,
        "gradient_clip_norm": 5.0,
        "internal_validation_fraction": 0.15,
        "device": device,
    }


def _optimizer_settings(
    model: str,
    parameters: Mapping[str, object],
) -> tuple[float, float, int]:
    if model == "mlp":
        return (
            float(parameters["learning_rate_init"]),
            float(parameters["alpha"]),
            int(parameters["batch_size"]),
        )
    return (
        float(parameters["learning_rate"]),
        float(parameters["weight_decay"]),
        int(parameters["batch_size"]),
    )


def _run_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    scheduler_horizon: int | None = None,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> None:
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(20, scheduler_horizon if scheduler_horizon is not None else epochs),
        eta_min=learning_rate * 0.05,
    )
    loss_function = nn.MSELoss()
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = generator.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            batch = torch.as_tensor(
                order[start : start + batch_size], dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            loss = loss_function(model(features[batch]), targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()


def _select_epoch(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    parameters: Mapping[str, object],
    *,
    seed: int,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> int:
    permutation = np.random.default_rng(seed).permutation(len(y))
    validation_size = max(1, int(round(len(y) * 0.15)))
    validation_index = np.sort(permutation[:validation_size])
    train_index = np.sort(permutation[validation_size:])
    x_scaler = StandardScaler().fit(x[train_index])
    y_scaler = StandardScaler().fit(y[train_index].reshape(-1, 1))
    train_x = x_scaler.transform(x[train_index])
    train_y = y_scaler.transform(y[train_index].reshape(-1, 1)).reshape(-1)
    valid_x = torch.as_tensor(
        x_scaler.transform(x[validation_index]), dtype=torch.float32, device=device
    )
    valid_y = torch.as_tensor(
        y_scaler.transform(y[validation_index].reshape(-1, 1)),
        dtype=torch.float32,
        device=device,
    )

    _set_seed(seed)
    model = _make_torch_model(model_name, x.shape[1], parameters).to(device)
    learning_rate, weight_decay, batch_size = _optimizer_settings(model_name, parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(20, max_epochs), eta_min=learning_rate * 0.05
    )
    loss_function = nn.MSELoss()
    features = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    targets = torch.as_tensor(train_y.reshape(-1, 1), dtype=torch.float32, device=device)
    generator = np.random.default_rng(seed + 1)
    best_loss = float("inf")
    selected_epoch = 1
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = generator.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            batch = torch.as_tensor(
                order[start : start + batch_size], dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            loss = loss_function(model(features[batch]), targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(valid_x), valid_y))
        if best_loss - validation_loss > min_delta:
            best_loss = validation_loss
            selected_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= min_epochs and stale_epochs >= patience:
            break
    return selected_epoch


def fit_final_model(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    parameters: Mapping[str, object],
    *,
    seed: int,
    max_epochs: int | None = None,
    min_epochs: int | None = None,
    patience: int | None = None,
    mlp_max_epochs: int | None = None,
    mlp_patience: int | None = None,
    device: str = "cpu",
) -> FittedSearchModel:
    """Fit one final model without accepting prediction-set labels."""

    key = _normalise_model_name(model)
    features = np.asarray(x, dtype=float)
    targets = np.asarray(y, dtype=float).reshape(-1)
    if features.ndim != 2 or features.shape[0] != targets.size or targets.size < 4:
        raise ValueError("x and y must contain at least four paired samples")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("x and y must contain only finite values")
    decoded = decode_hyperparameters(key, parameters)
    x_scaler = StandardScaler().fit(features)
    y_scaler = StandardScaler().fit(targets.reshape(-1, 1))
    scaled_x = x_scaler.transform(features)
    scaled_y = y_scaler.transform(targets.reshape(-1, 1)).reshape(-1)

    if key == "svr":
        estimator = SVR(
            kernel="rbf",
            C=float(decoded["C"]),
            gamma=float(decoded["gamma"]),
            epsilon=float(decoded["epsilon"]),
        )
        estimator.fit(scaled_x, scaled_y)
        return FittedSearchModel(key, estimator, x_scaler, y_scaler)

    if key == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            **decoded,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        estimator.fit(scaled_x, scaled_y)
        return FittedSearchModel(key, estimator, x_scaler, y_scaler)

    resolved_max, resolved_min, resolved_patience, min_delta = _training_schedule(
        key,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        patience=patience,
        mlp_max_epochs=mlp_max_epochs,
        mlp_patience=mlp_patience,
    )
    torch_device = torch.device(device)
    selected_epoch = _select_epoch(
        key,
        features,
        targets,
        decoded,
        seed=seed,
        max_epochs=resolved_max,
        min_epochs=resolved_min,
        patience=resolved_patience,
        min_delta=min_delta,
        device=torch_device,
    )
    _set_seed(seed)
    estimator = _make_torch_model(key, features.shape[1], decoded).to(torch_device)
    learning_rate, weight_decay, batch_size = _optimizer_settings(key, decoded)
    _run_epochs(
        estimator,
        scaled_x,
        scaled_y,
        epochs=selected_epoch,
        scheduler_horizon=resolved_max,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed + 2,
        device=torch_device,
    )
    estimator.eval()
    return FittedSearchModel(
        key,
        estimator,
        x_scaler,
        y_scaler,
        selected_epoch=selected_epoch,
        device=str(torch_device),
    )


def make_fit_predict(
    model: str,
    *,
    max_epochs: int | None = None,
    min_epochs: int | None = None,
    patience: int | None = None,
    mlp_max_epochs: int | None = None,
    mlp_patience: int | None = None,
    device: str = "cpu",
) -> Callable[..., np.ndarray]:
    """Return the fold evaluator consumed by :func:`bayesian_search`."""

    key = _normalise_model_name(model)

    def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        parameters: Mapping[str, object],
        *,
        fold_seed: int,
    ) -> np.ndarray:
        fitted = fit_final_model(
            key,
            x_train,
            y_train,
            parameters,
            seed=fold_seed,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            patience=patience,
            mlp_max_epochs=mlp_max_epochs,
            mlp_patience=mlp_patience,
            device=device,
        )
        return fitted.predict(x_valid)

    return fit_predict
