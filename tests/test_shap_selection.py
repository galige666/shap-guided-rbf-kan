from __future__ import annotations

import numpy as np

from shap_rbf_kan.shap_selection import mean_absolute_shap, select_top_k


def test_mean_absolute_shap_supports_single_and_multiple_outputs():
    single = np.array([[1.0, -2.0, 3.0], [-3.0, 4.0, -1.0]])
    multiple = np.stack([single, single * 2.0], axis=-1)

    np.testing.assert_allclose(mean_absolute_shap(single), [2.0, 3.0, 2.0])
    np.testing.assert_allclose(mean_absolute_shap(multiple), [3.0, 4.5, 3.0])


def test_top_k_uses_stable_feature_order_for_ties():
    importance = np.array([0.5, 0.8, 0.8, 0.2])

    selected = select_top_k(importance, 3)

    np.testing.assert_array_equal(selected, [1, 2, 0])
