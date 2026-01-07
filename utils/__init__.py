"""
Utility Functions

This package contains various helper utilities for CRATE,
including metrics and tools such as coding rate computation.
"""

from .metrics import CodingRate, cal_sparsity
from .hook_manager import remove_all_hooks, CRATEHookManager
from .visualization import visualize_model_graph

__all__ = [
    "CodingRate",
    "cal_sparsity",
    "remove_all_hooks",
    "CRATEHookManager",
    "visualize_model_graph",
]
