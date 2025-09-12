import cupy as cp
import os

def _load_find_closest_saturated_nodes_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "find_closest_saturated_nodes.cu"), "r") as f:
        find_closest_saturated_nodes = f.read()
    return cp.RawKernel(find_closest_saturated_nodes, "find_closest_saturated_nodes")

def _load_get_nhood_idx_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "get_nhood_idx.cu"), "r") as f:
        get_nhood_idx = f.read()
    return cp.RawKernel(get_nhood_idx, "get_nhood_idx")

def _load_get_nhood_idx_with_distance_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "get_nhood_idx_with_distance.cu"), "r") as f:
        get_nhood_idx_with_distance_code = f.read()
    return cp.RawKernel(get_nhood_idx_with_distance_code, "get_nhood_idx_with_distance")

def _load_sepal_simulation_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "sepal_simulation.cu"), "r") as f:
        code = f.read()
    step_kernel = cp.RawKernel(code, "sepal_simulation")
    return step_kernel

def _load_sepal_simulation_debug_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "sepal_simulation_debug.cu"), "r") as f:
        code = f.read()
    step_kernel = cp.RawKernel(code, "sepal_simulation_debug")
    return step_kernel


sepal_simulation_debug = _load_sepal_simulation_debug_kernel()
sepal_simulation = _load_sepal_simulation_kernel()
find_closest_saturated_nodes = _load_find_closest_saturated_nodes_kernel()
get_nhood_idx = _load_get_nhood_idx_kernel()
get_nhood_idx_with_distance = _load_get_nhood_idx_with_distance_kernel()