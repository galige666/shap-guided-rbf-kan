from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np


def _spectral_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("spectra must have shape (samples, features)")
    if matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("spectra must contain at least two finite features")
    return matrix


def baseline_correction(spectra: np.ndarray, *, degree: int = 2) -> np.ndarray:
    """Remove a per-spectrum polynomial baseline."""

    matrix = _spectral_matrix(spectra)
    if degree < 0 or degree >= matrix.shape[1]:
        raise ValueError("degree must be non-negative and smaller than the feature count")
    coordinate = np.linspace(-1.0, 1.0, matrix.shape[1])
    corrected = np.empty_like(matrix)
    for index, spectrum in enumerate(matrix):
        baseline = np.polyval(np.polyfit(coordinate, spectrum, degree), coordinate)
        corrected[index] = spectrum - baseline
    return corrected


def standard_normal_variate(spectra: np.ndarray) -> np.ndarray:
    """Apply row-wise mean centring and sample-standard-deviation scaling."""

    matrix = _spectral_matrix(spectra)
    means = matrix.mean(axis=1, keepdims=True)
    scales = matrix.std(axis=1, ddof=1, keepdims=True)
    scales = np.where(scales > np.finfo(float).eps, scales, 1.0)
    return (matrix - means) / scales


def _checked_window(feature_count: int, window_length: int, polyorder: int) -> int:
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and greater than polyorder")
    if window_length > feature_count:
        raise ValueError("window_length cannot exceed the feature count")
    return int(window_length)


def _savgol(
    matrix: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
    derivative: int,
) -> np.ndarray:
    window = _checked_window(matrix.shape[1], window_length, polyorder)
    if derivative < 0 or derivative > polyorder:
        raise ValueError("derivative must be between zero and polyorder")
    half_window = window // 2
    coordinates = np.arange(-half_window, half_window + 1, dtype=float)
    design = np.vander(coordinates, polyorder + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[derivative] * math.factorial(derivative)
    padded = np.pad(matrix, ((0, 0), (half_window, half_window)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    return np.einsum("snw,w->sn", windows, coefficients)


def savitzky_golay(
    spectra: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    matrix = _spectral_matrix(spectra)
    return _savgol(
        matrix,
        window_length=window_length,
        polyorder=polyorder,
        derivative=0,
    )


def first_derivative(
    spectra: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    matrix = _spectral_matrix(spectra)
    return _savgol(
        matrix,
        window_length=window_length,
        polyorder=polyorder,
        derivative=1,
    )


def second_derivative(
    spectra: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    matrix = _spectral_matrix(spectra)
    return _savgol(
        matrix,
        window_length=window_length,
        polyorder=polyorder,
        derivative=2,
    )


def apply_pipeline(spectra: np.ndarray, operations: Iterable[str]) -> np.ndarray:
    """Apply named preprocessing operations in the supplied order."""

    result = _spectral_matrix(spectra).copy()
    functions = {
        "bc": baseline_correction,
        "snv": standard_normal_variate,
        "sg": savitzky_golay,
        "d1": first_derivative,
        "d2": second_derivative,
    }
    for operation in operations:
        key = operation.strip().lower()
        if key not in functions:
            raise ValueError(f"unknown preprocessing operation: {operation}")
        result = functions[key](result)
    return result
