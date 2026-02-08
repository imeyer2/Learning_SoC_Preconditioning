"""
PyAMG Interface Module.

Handles interaction with PyAMG: solver building, C matrix construction,
edge sampling strategies.
"""

from .solver_builder import (
    build_pyamg_solver,
    build_C_from_model,
    C_from_selected_edges,
    build_B_for_pyamg,
    print_hierarchy_info,
    get_hierarchy_summary,
)
from .sampling import (
    sample_topk_without_replacement,
    sample_deterministic_topk,
    build_row_groups,
)

__all__ = [
    "build_pyamg_solver",
    "build_C_from_model",
    "C_from_selected_edges",
    "build_B_for_pyamg",
    "print_hierarchy_info",
    "get_hierarchy_summary",
    "sample_topk_without_replacement",
    "sample_deterministic_topk",
    "build_row_groups",
]
