from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _BasisEdgeLayer(nn.Module):
    """Shared base and learned-basis branches used by the paper's KAN layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or num_basis < 2:
            raise ValueError("in_features and out_features must be positive; num_basis must be >= 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_basis = int(num_basis)
        self.layer_norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()
        self.base_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.basis_weight = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    @property
    def coefficients(self) -> nn.Parameter:
        """Backward-compatible name for the learned basis weights."""

        return self.basis_weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.trunc_normal_(self.basis_weight, std=0.03)
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f"inputs must have shape (samples, {self.in_features})")
        normalized = self.layer_norm(inputs)
        bases = self.dropout(self.basis_functions(normalized))
        basis_output = torch.einsum("nib,oib->no", bases, self.basis_weight)
        base_output = F.linear(self.base_activation(normalized), self.base_weight)
        return basis_output + base_output + self.bias


class GaussianEdgeLayer(_BasisEdgeLayer):
    """KAN layer with trainable base branches and fixed Gaussian bases."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        width: float | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        super().__init__(
            in_features,
            out_features,
            num_basis=num_basis,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        centres = torch.linspace(grid_min, grid_max, num_basis)
        spacing = float((grid_max - grid_min) / (num_basis - 1))
        self.register_buffer("centres", centres)
        self.width = float(width if width is not None else spacing)
        if self.width <= 0:
            raise ValueError("width must be positive")

    def basis_functions(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f"inputs must have shape (samples, {self.in_features})")
        normalized = (inputs.unsqueeze(-1) - self.centres) / self.width
        return torch.exp(-(normalized.square()))


class BSplineEdgeLayer(_BasisEdgeLayer):
    """KAN layer using centered local cubic B-spline bases on every edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_basis: int,
        degree: int = 3,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        if degree != 3:
            raise ValueError("the reference implementation uses cubic (degree=3) bases")
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        super().__init__(
            in_features,
            out_features,
            num_basis=num_basis,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.degree = int(degree)
        self.register_buffer("centres", torch.linspace(grid_min, grid_max, num_basis))
        self.width = float((grid_max - grid_min) / (num_basis - 1))

    def basis_functions(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f"inputs must have shape (samples, {self.in_features})")
        distance = torch.abs((inputs.unsqueeze(-1) - self.centres) / self.width)
        inner = (2.0 / 3.0) - distance.square() + 0.5 * distance.pow(3)
        outer = (2.0 - distance).clamp_min(0).pow(3) / 6.0
        return torch.where(
            distance < 1.0,
            inner,
            torch.where(distance < 2.0, outer, torch.zeros_like(distance)),
        )


class _BasisKANRegressor(nn.Module):
    layer_type: type[nn.Module]

    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: Sequence[int] = (16,),
        num_basis: int = 5,
        grid_min: float = -3.0,
        grid_max: float = 3.0,
        dropout: float = 0.0,
        **basis_options: object,
    ) -> None:
        super().__init__()
        dimensions = [int(in_features), *(int(value) for value in hidden_features), 1]
        if any(value < 1 for value in dimensions):
            raise ValueError("all layer dimensions must be positive")
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(
                self.layer_type(
                    source,
                    target,
                    num_basis=num_basis,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    dropout=dropout if index < len(dimensions) - 2 else 0.0,
                    use_layer_norm=index > 0,
                    **basis_options,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


class RBFKANRegressor(_BasisKANRegressor):
    layer_type = GaussianEdgeLayer


class BSplineKANRegressor(_BasisKANRegressor):
    layer_type = BSplineEdgeLayer


class SpectralCNN(nn.Module):
    """Small 1D CNN used as a nonlinear spectral-regression baseline."""

    def __init__(
        self,
        *,
        channels: int = 16,
        kernel_size: int = 9,
        dense_features: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(channels, channels * 2, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2, dense_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_features, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("inputs must have shape (samples, features) or (samples, 1, features)")
        return self.regressor(self.features(inputs))
