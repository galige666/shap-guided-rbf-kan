from __future__ import annotations

import numpy as np
import pytest

from shap_rbf_kan.preprocessing import (
    apply_pipeline,
    first_derivative,
    savitzky_golay,
    second_derivative,
    standard_normal_variate,
)


def test_snv_centres_and_scales_each_spectrum():
    spectra = np.array([[1.0, 2.0, 4.0, 7.0], [3.0, 5.0, 9.0, 15.0]])
    transformed = standard_normal_variate(spectra)

    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=1, ddof=1), 1.0, atol=1e-12)


@pytest.mark.parametrize("operation", [savitzky_golay, first_derivative, second_derivative])
def test_savgol_operations_preserve_shape_and_finite_values(operation):
    x = np.linspace(0.0, 2.0 * np.pi, 21)
    spectra = np.vstack([np.sin(x), np.cos(x)])

    transformed = operation(spectra, window_length=7, polyorder=3)

    assert transformed.shape == spectra.shape
    assert np.isfinite(transformed).all()


def test_pipeline_rejects_unknown_operation():
    with pytest.raises(ValueError, match="unknown preprocessing operation"):
        apply_pipeline(np.ones((2, 9)), ["snv", "mystery"])
