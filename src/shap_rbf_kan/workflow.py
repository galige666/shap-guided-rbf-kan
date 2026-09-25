from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .models import RBFKANRegressor
from .shap_selection import mean_absolute_shap, select_top_k


@dataclass(frozen=True)
class ArrayScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayScaler":
        array = np.asarray(values, dtype=float)
        mean = array.mean(axis=0)
        scale = array.std(axis=0, ddof=0)
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        return cls(np.asarray(mean), np.asarray(scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


@dataclass
class FittedRBFKAN:
    model: RBFKANRegressor
    x_scaler: ArrayScaler
    y_scaler: ArrayScaler
    selected_epoch: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    config: dict[str, object]

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        transformed = torch.as_tensor(self.x_scaler.transform(x), dtype=torch.float32)
        with torch.no_grad():
            scaled = self.model(transformed).cpu().numpy().reshape(-1)
        return self.y_scaler.inverse_transform(scaled).reshape(-1)


@dataclass
class WorkflowResult:
    full_model: FittedRBFKAN
    compact_model: FittedRBFKAN
    feature_importance: np.ndarray
    selected_features: np.ndarray
    model_feature_indices: np.ndarray
    full_predictions: np.ndarray
    compact_predictions: np.ndarray
    full_metrics: dict[str, float] | None
    compact_metrics: dict[str, float] | None


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_model(feature_count: int, config: Mapping[str, object]) -> RBFKANRegressor:
    half_width = float(config.get("grid_half_width", 3.0))
    return RBFKANRegressor(
        feature_count,
        hidden_features=tuple(int(value) for value in config.get("hidden_features", [16])),
        num_basis=int(config.get("num_basis", 5)),
        grid_min=-half_width,
        grid_max=half_width,
        dropout=float(config.get("dropout", 0.0)),
    )


def _train_fixed_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> None:
    features = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(20, epochs), eta_min=learning_rate * 0.05
    )
    loss_function = nn.MSELoss()
    generator = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = generator.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            batch = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_function(model(features[batch]), targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()


def fit_rbf_kan(
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    config: Mapping[str, object],
    *,
    validation_fraction: float = 0.15,
    max_epochs: int = 700,
    min_epochs: int = 80,
    patience: int = 80,
    min_delta: float = 1e-7,
) -> FittedRBFKAN:
    """Select an epoch internally, then refit on the complete calibration set."""

    x = np.asarray(calibration_x, dtype=float)
    y = np.asarray(calibration_y, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] < 4:
        raise ValueError("calibration_x and calibration_y must contain at least four paired samples")
    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction must be between zero and 0.5")
    if min_epochs < 1 or max_epochs < min_epochs or patience < 1:
        raise ValueError("epoch and patience settings are inconsistent")

    seed = int(config.get("seed", 11))
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0001))
    batch_size = int(config.get("batch_size", 64))
    permutation = np.random.default_rng(seed).permutation(x.shape[0])
    validation_size = max(1, int(round(x.shape[0] * validation_fraction)))
    validation_indices = np.sort(permutation[:validation_size])
    train_indices = np.sort(permutation[validation_size:])

    inner_x_scaler = ArrayScaler.fit(x[train_indices])
    inner_y_scaler = ArrayScaler.fit(y[train_indices])
    train_x = inner_x_scaler.transform(x[train_indices])
    train_y = inner_y_scaler.transform(y[train_indices]).reshape(-1)
    valid_x = torch.as_tensor(inner_x_scaler.transform(x[validation_indices]), dtype=torch.float32)
    valid_y = torch.as_tensor(
        inner_y_scaler.transform(y[validation_indices]).reshape(-1, 1), dtype=torch.float32
    )

    _set_seed(seed)
    selection_model = _make_model(x.shape[1], config)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(20, max_epochs), eta_min=learning_rate * 0.05
    )
    loss_function = nn.MSELoss()
    features = torch.as_tensor(train_x, dtype=torch.float32)
    targets = torch.as_tensor(train_y.reshape(-1, 1), dtype=torch.float32)
    generator = np.random.default_rng(seed + 1)
    best_loss = float("inf")
    selected_epoch = 1
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        selection_model.train()
        order = generator.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            batch = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_function(selection_model(features[batch]), targets[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selection_model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
        selection_model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(selection_model(valid_x), valid_y))
        if best_loss - validation_loss > min_delta:
            best_loss = validation_loss
            selected_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= min_epochs and stale_epochs >= patience:
            break

    final_x_scaler = ArrayScaler.fit(x)
    final_y_scaler = ArrayScaler.fit(y)
    _set_seed(seed)
    final_model = _make_model(x.shape[1], config)
    _train_fixed_epochs(
        final_model,
        final_x_scaler.transform(x),
        final_y_scaler.transform(y).reshape(-1),
        epochs=selected_epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed + 2,
    )
    final_model.eval()
    return FittedRBFKAN(
        model=final_model,
        x_scaler=final_x_scaler,
        y_scaler=final_y_scaler,
        selected_epoch=selected_epoch,
        train_indices=train_indices,
        validation_indices=validation_indices,
        config=dict(config),
    )


def gradient_shap_values(
    fitted: FittedRBFKAN,
    calibration_x: np.ndarray,
    *,
    background_size: int = 24,
    draws: int = 8,
) -> np.ndarray:
    """Approximate Gradient SHAP with expected gradients over calibration backgrounds."""

    transformed = fitted.x_scaler.transform(calibration_x)
    if transformed.shape[0] < 2:
        raise ValueError("at least two calibration samples are required for Gradient SHAP")
    rng = np.random.default_rng(int(fitted.config.get("seed", 11)) + 503)
    background_indices = rng.choice(
        transformed.shape[0], size=min(background_size, transformed.shape[0]), replace=False
    )
    background = transformed[background_indices]
    values = np.zeros_like(transformed)
    fitted.model.eval()
    for sample_index, sample in enumerate(transformed):
        contributions = []
        for _ in range(draws):
            reference = background[rng.integers(0, len(background))]
            alpha = rng.random()
            interpolated = reference + alpha * (sample - reference)
            tensor = torch.tensor(interpolated[None, :], dtype=torch.float32, requires_grad=True)
            fitted.model.zero_grad(set_to_none=True)
            fitted.model(tensor).sum().backward()
            contributions.append((sample - reference) * tensor.grad.detach().cpu().numpy()[0])
        values[sample_index] = np.mean(contributions, axis=0)
    return values


def _regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    residual = prediction - reference
    total = float(np.sum((reference - reference.mean()) ** 2))
    return {
        "r2": float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def run_shap_guided_workflow(
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    *,
    config: Mapping[str, object],
    top_k: int,
    shap_values: np.ndarray | None = None,
    prediction_x: np.ndarray | None = None,
    prediction_y: np.ndarray | None = None,
    max_epochs: int = 700,
    min_epochs: int = 80,
    patience: int = 80,
    min_delta: float = 1e-7,
) -> WorkflowResult:
    """Fit full and SHAP-selected models using calibration-only development."""

    x = np.asarray(calibration_x, dtype=float)
    supplied_attributions = None
    if shap_values is not None:
        supplied_attributions = np.asarray(shap_values, dtype=float)
        valid_shape = (
            supplied_attributions.ndim in (2, 3)
            and supplied_attributions.shape[0] == x.shape[0]
            and supplied_attributions.shape[1] == x.shape[1]
        )
        if not valid_shape:
            raise ValueError(
                "SHAP values must have shape "
                f"({x.shape[0]}, {x.shape[1]}) or ({x.shape[0]}, {x.shape[1]}, outputs)"
            )
    fit_options = {
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "min_delta": min_delta,
    }
    full_model = fit_rbf_kan(x, calibration_y, config, **fit_options)
    attributions = (
        supplied_attributions
        if supplied_attributions is not None
        else gradient_shap_values(full_model, x)
    )
    importance = mean_absolute_shap(attributions)
    selected = select_top_k(importance, top_k)
    model_feature_indices = np.sort(selected)
    compact_model = fit_rbf_kan(
        x[:, model_feature_indices], calibration_y, config, **fit_options
    )

    if prediction_x is None:
        full_predictions = np.empty(0, dtype=float)
        compact_predictions = np.empty(0, dtype=float)
    else:
        prediction_features = np.asarray(prediction_x, dtype=float)
        if prediction_features.ndim != 2 or prediction_features.shape[1] != x.shape[1]:
            raise ValueError(
                f"prediction_x must have shape (samples, {x.shape[1]})"
            )
        full_predictions = full_model.predict(prediction_features)
        compact_predictions = compact_model.predict(
            prediction_features[:, model_feature_indices]
        )

    if prediction_y is not None:
        if prediction_x is None:
            raise ValueError("prediction_x is required when prediction_y is supplied")
        full_metrics = _regression_metrics(prediction_y, full_predictions)
        compact_metrics = _regression_metrics(prediction_y, compact_predictions)
    else:
        full_metrics = None
        compact_metrics = None

    return WorkflowResult(
        full_model=full_model,
        compact_model=compact_model,
        feature_importance=importance,
        selected_features=selected,
        model_feature_indices=model_feature_indices,
        full_predictions=full_predictions,
        compact_predictions=compact_predictions,
        full_metrics=full_metrics,
        compact_metrics=compact_metrics,
    )
