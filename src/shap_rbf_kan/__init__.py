"""Compact reference implementation of the SHAP-guided RBF-KAN workflow.

Modules are intentionally imported on demand so scientific libraries can manage
their native runtimes without package-import side effects.
"""

__version__ = "0.3.0"
__all__ = [
    "model_search",
    "models",
    "preprocessing",
    "shap_selection",
    "tuning",
    "workflow",
]
