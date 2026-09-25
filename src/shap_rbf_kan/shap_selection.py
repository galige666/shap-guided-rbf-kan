from __future__ import annotations

import numpy as np


def mean_absolute_shap(values: np.ndarray) -> np.ndarray:
    """Return one mean absolute SHAP importance value per input feature."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        importance = np.mean(np.abs(array), axis=0)
    elif array.ndim == 3:
        importance = np.mean(np.abs(array), axis=(0, 2))
    else:
        raise ValueError("SHAP values must have shape (samples, features[, outputs])")
    if not np.isfinite(importance).all():
        raise ValueError("SHAP values must be finite")
    return importance


def select_top_k(values: np.ndarray, k: int) -> np.ndarray:
    """Select feature indices by descending importance with stable tie handling."""

    array = np.asarray(values, dtype=float)
    importance = array if array.ndim == 1 else mean_absolute_shap(array)
    if importance.ndim != 1 or not np.isfinite(importance).all():
        raise ValueError("importance must be a finite one-dimensional array")
    if not 1 <= k <= importance.size:
        raise ValueError("k must be between 1 and the number of features")
    return np.argsort(-importance, kind="stable")[:k]
