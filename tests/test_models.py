from __future__ import annotations

import pytest
import torch
from torch import nn

from shap_rbf_kan.models import (
    BSplineEdgeLayer,
    BSplineKANRegressor,
    RBFKANRegressor,
    SpectralCNN,
)


@pytest.mark.parametrize("model_class", [RBFKANRegressor, BSplineKANRegressor])
def test_kan_regressors_produce_scalar_predictions_and_gradients(model_class):
    torch.manual_seed(7)
    model = model_class(8, hidden_features=(5,), num_basis=5)
    inputs = torch.randn(6, 8)

    predictions = model(inputs)
    predictions.mean().backward()

    assert predictions.shape == (6, 1)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_cubic_bspline_basis_is_finite_nonnegative_and_has_requested_size():
    layer = BSplineEdgeLayer(3, 2, num_basis=6, degree=3, grid_min=-2.0, grid_max=2.0)
    values = layer.basis_functions(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 0.5, 1.5]]))

    assert values.shape == (2, 3, 6)
    assert torch.isfinite(values).all()
    assert torch.all(values >= 0)


def test_basis_layers_match_the_paper_architecture_and_allow_three_basis_functions():
    layer = BSplineEdgeLayer(3, 2, num_basis=3, degree=3, grid_min=-2.0, grid_max=2.0)
    values = layer(torch.tensor([[-2.0, 0.0, 2.0], [-1.0, 0.5, 1.5]]))

    assert values.shape == (2, 2)
    assert hasattr(layer, "base_weight")
    assert hasattr(layer, "basis_weight")

    model = RBFKANRegressor(8, hidden_features=(5, 4), num_basis=3, dropout=0.1)
    assert isinstance(model.layers[0].layer_norm, nn.Identity)
    assert isinstance(model.layers[1].layer_norm, nn.LayerNorm)
    assert isinstance(model.layers[0].dropout, nn.Dropout)
    assert isinstance(model.layers[-1].dropout, nn.Identity)


def test_spectral_cnn_accepts_flat_spectra_and_backpropagates():
    torch.manual_seed(9)
    model = SpectralCNN(channels=4, kernel_size=3, dense_features=6, dropout=0.1)
    inputs = torch.randn(6, 32)

    predictions = model(inputs)
    predictions.square().mean().backward()

    assert predictions.shape == (6, 1)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
