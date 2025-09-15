from __future__ import annotations

import os

import cupy as cp


def _load_get_nhood_idx_with_distance_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "get_nhood_idx_with_distance.cu")) as f:
        get_nhood_idx_with_distance_code = f.read()
    return cp.RawKernel(get_nhood_idx_with_distance_code, "get_nhood_idx_with_distance")


def _load_sepal_simulation_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "sepal_simulation.cu")) as f:
        code = f.read()
    # Load the new multi-gene batch kernel
    step_kernel = cp.RawKernel(code, "sepal_simulation")
    return step_kernel


sepal_simulation = _load_sepal_simulation_kernel()
get_nhood_idx_with_distance = _load_get_nhood_idx_with_distance_kernel()
