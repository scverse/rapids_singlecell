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

def _load_assign_closest_saturated_kernel():
    kernel_dir = os.path.dirname(__file__)
    with open(os.path.join(kernel_dir, "assign_closest_saturated.cu"), "r") as f:
        assign_closest_saturated = f.read()
    return cp.RawKernel(assign_closest_saturated, "assign_closest_saturated_to_unresolved")

find_closest_saturated_nodes = _load_find_closest_saturated_nodes_kernel()
get_nhood_idx = _load_get_nhood_idx_kernel()
assign_closest_saturated_kernel = _load_assign_closest_saturated_kernel()