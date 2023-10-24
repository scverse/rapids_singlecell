from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData


def read_mtx(
    filename,
    backend: Literal["cudf", "dask_cudf"] = "cudf",
    output: Literal["CPU", "GPU"] = "CPU",
):
    """
    Read mtx using using GPU, the matrix is transposed by default

    Parameters
    ----------
    filename
        Name of the matrix file, in mtx or compressed gz format.
    backend
        Which backend to use, `dask_cudf` comes handy when there is not enough GPU memory, in such case the output will be automatically sent to CPU.
    output
        Where to keep the matrix, either keep to the GPU memory, or send it to RAM.
    """
    import cupyx.scipy.sparse as csp
    import scipy.sparse as sp

    mtxinfo = pd.read_csv(filename, nrows=1, sep=" ", comment="%", header=None).values[
        0
    ]
    shape = tuple((mtxinfo[[1, 0]]).astype(int))

    if backend == "cudf":
        import cudf

        mtx_data = cudf.read_csv(
            filename,
            sep=" ",
            dtype=["float32" for i in range(3)],
            comment="%",
            header=None,
            skiprows=2,
        )
        # offseting row and column indices to fit python indexing
        mtx_data["0"] = mtx_data["0"] - 1
        mtx_data["1"] = mtx_data["1"] - 1

        mtx_data = mtx_data.to_cupy()

        mtx_data = csp.coo_matrix(
            (mtx_data[:, 2], (mtx_data[:, 1], mtx_data[:, 0])),
            shape=shape,
            dtype=np.float32,
        )
        toadata = mtx_data.get().tocsr() if output == "CPU" else mtx_data.tocsr()

    elif backend == "dask_cudf":
        import dask_cudf

        output = "CPU"
        mtx_data = dask_cudf.read_csv(
            filename,
            sep=" ",
            dtype=["float32" for i in range(3)],
            comment="%",
            header=None,
        )
        mtx_data = mtx_data.to_dask_dataframe()  # loading back to host
        toadata = sp.coo_matrix(
            (mtx_data["2"][1:], (mtx_data["1"][1:] - 1, mtx_data["0"][1:] - 1)),
            shape=shape,
            dtype=np.float32,
        )
        toadata = toadata.tocsr()

    return AnnData(toadata)
