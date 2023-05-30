import cupy as cp
import cupyx as cpx
import math
import warnings
from typing import Optional

from ..cunnData import cunnData
from ._utils import _check_nonnegative_integers


def normalize_total(
    cudata: cunnData, target_sum: int, layer: Optional[str] = None, inplace: bool = True
) -> Optional[cpx.scipy.sparse.csr_matrix]:
    """
    Normalizes rows in matrix so they sum to `target_sum`

    Parameters
    ----------
        cudata:
            cunnData object

        target_sum :
            Each row will be normalized to sum to this value

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        inplace
            Whether to update `cudata` or return the normalized matrix.


    Returns
    -------
    Returns a normalized copy or  updates `cudata` with a normalized version of \
    the original `cudata.X` and `cudata.layers['layer']`, depending on `inplace`.

    """
    csr_arr = cudata.layers[layer] if layer is not None else cudata.X

    if not inplace:
        csr_arr = csr_arr.copy()

    mul_kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void mul_kernel(const int *indptr, float *data,
                        int nrows, int tsum) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;

            if(row >= nrows)
                return;

            float scale = 0.0;
            int start_idx = indptr[row];
            int stop_idx = indptr[row+1];

            for(int i = start_idx; i < stop_idx; i++)
                scale += data[i];

            if(scale > 0.0) {
                scale = tsum / scale;
                for(int i = start_idx; i < stop_idx; i++)
                    data[i] *= scale;
            }
        }
        """,
        "mul_kernel",
    )

    mul_kernel(
        (math.ceil(csr_arr.shape[0] / 32.0),),
        (32,),
        (csr_arr.indptr, csr_arr.data, csr_arr.shape[0], int(target_sum)),
    )

    if inplace:
        if layer:
            cudata.layers[layer] = csr_arr
        else:
            cudata.X = csr_arr
    else:
        return csr_arr


def log1p(
    cudata: cunnData, layer: Optional[str] = None, copy: bool = False
) -> Optional[cpx.scipy.sparse.spmatrix]:
    """
    Calculated the natural logarithm of one plus the sparse matrix.

    Parameters
    ----------
        cudata
            cunnData object

        layer
            Layer to normalize instead of `X`. If `None`, `X` is normalized.

        copy
            Whether to return a copy or update `cudata`.

    Returns
    ----------
            The resulting sparse matrix after applying the natural logarithm of one plus the input matrix. \
            If `copy` is set to True, returns the new sparse matrix. Otherwise, updates the `cudata` object \
            in-place and returns None.

    """
    X = cudata.layers[layer] if layer is not None else cudata.X
    X = X.log1p()
    cudata.uns["log1p"] = {"base": None}
    if not copy:
        if layer:
            cudata.layers[layer] = X
        else:
            cudata.X = X
    else:
        return X


_sparse_kernel_sum_csc = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_sum_cg_csc(const int *indptr,const int *index,const float *data,
                                        float* sums_genes, float* sums_cells,
                                        int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene+1];

        for(int cell = start_idx; cell < stop_idx; cell++){
            float value = data[cell];
            int cell_number = index[cell];
            atomicAdd(&sums_genes[gene], value);
            atomicAdd(&sums_cells[cell_number], value);

        }
    }
    """,
    "calculate_sum_cg_csc",
)

_sparse_kernel_norm_res_csc = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_res_csc(const int *indptr,const int *index,const float *data,
                            const float* sums_cells,const float* sums_genes,
                            float* residuals ,const float* sum_total, const float* clip,
                            const float* theta,const int n_cells,const int n_genes) {
        int gene = blockDim.x * blockIdx.x + threadIdx.x;
        if(gene >= n_genes){
            return;
        }
        int start_idx = indptr[gene];
        int stop_idx = indptr[gene + 1];

        int sparse_idx = start_idx;
        for(int cell = 0; cell < n_cells; cell++){
            float mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
            long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
            if (sparse_idx < stop_idx && index[sparse_idx] == cell){
                residuals[res_index] += data[sparse_idx];
                sparse_idx++;
            }
            residuals[res_index] -= mu;
            residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
            residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
        }
    }
    """,
    "calculate_res_csc",
)


_sparse_kernel_sum_csr = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_sum_cg_csr(const int *indptr,const int *index,const float *data,
                                        float* sums_genes, float* sums_cells,
                                        int n_cells) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell + 1];

        for(int gene = start_idx; gene < stop_idx; gene++){
            float value = data[gene];
            int gene_number = index[gene];
            atomicAdd(&sums_genes[gene_number], value);
            atomicAdd(&sums_cells[cell], value);

        }
    }
    """,
    "calculate_sum_cg_csr",
)

_sparse_kernel_norm_res_csr = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_res_csr(const int * indptr, const int * index, const float * data,
                       const float * sums_cells, const float * sums_genes,
                       float * residuals, const float * sum_total, const float * clip,
                       const float * theta, const int n_cells, const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        if(cell >= n_cells){
            return;
        }
        int start_idx = indptr[cell];
        int stop_idx = indptr[cell + 1];

        int sparse_idx = start_idx;
        for(int gene = 0; gene < n_genes; gene++){
            long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
            float mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
            if (sparse_idx < stop_idx && index[sparse_idx] == gene){
                residuals[res_index] += data[sparse_idx];
                sparse_idx++;
            }
            residuals[res_index] -= mu;
            residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
            residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
        }
    }
    """,
    "calculate_res_csr",
)


_dense_kernel_sum = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_sum_dense_cg(const float* residuals,
                        float* sums_cells,float* sums_genes,
                        const int n_cells,const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >= n_genes){
            return;
        }
        long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
        atomicAdd(&sums_genes[gene], residuals[res_index]);
        atomicAdd(&sums_cells[cell], residuals[res_index]);
    }
    """,
    "calculate_sum_dense_cg",
)


_kernel_norm_res_dense = cp.RawKernel(
    r"""
    extern "C" __global__
    void calculate_res_dense(const float* X,float* residuals,
                            const float* sums_cells,const float* sums_genes,
                            const float* sum_total,const float* clip,const float* theta,
                            const int n_cells, const int n_genes) {
        int cell = blockDim.x * blockIdx.x + threadIdx.x;
        int gene = blockDim.y * blockIdx.y + threadIdx.y;
        if(cell >= n_cells || gene >= n_genes){
            return;
        }

        float mu = sums_genes[gene]*sums_cells[cell]*sum_total[0];
        long long int res_index = static_cast<long long int>(cell) * n_genes + gene;
        residuals[res_index] = X[res_index] - mu;
        residuals[res_index] /= sqrt(mu + mu * mu * theta[0]);
        residuals[res_index]= fminf(fmaxf(residuals[res_index], -clip[0]), clip[0]);
    }
    """,
    "calculate_res_dense",
)


def normalize_pearson_residuals(
    cudata: cunnData,
    theta: float = 100,
    clip: Optional[float] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    inplace: bool = True,
) -> Optional[cp.ndarray]:
    """
    Applies analytic Pearson residual normalization, based on Lause21.
    The residuals are based on a negative binomial offset model with overdispersion
    `theta` shared across genes. By default, residuals are clipped to `sqrt(n_obs)`
    and overdispersion `theta=100` is used.

    Parameters
    ----------
        cudata
            cunnData object
        theta
            The negative binomial overdispersion parameter theta for Pearson residuals.
            Higher values correspond to less overdispersion (var = mean + mean^2/theta), and theta=np.Inf corresponds to a Poisson model.
        clip
            Determines if and how residuals are clipped:
            If None, residuals are clipped to the interval [-sqrt(n_obs), sqrt(n_obs)], where n_obs is the number of cells in the dataset (default behavior).
            If any scalar c, residuals are clipped to the interval [-c, c]. Set clip=np.Inf for no clipping.
        check_values
            If True, checks if counts in selected layer are integers as expected by this function,
            and return a warning if non-integers are found. Otherwise, proceed without checking. Setting this to False can speed up code for large datasets.
        layer
            Layer to use as input instead of X. If None, X is used.
        inplace
            If True, update cunnData with results. Otherwise, return results. See below for details of what is returned.

    Returns
    ----------
        If `inplace=True`, `cudata.X` or the selected layer in `cudata.layers` is updated with the normalized values. \
        If `inplace=False` the normalized matrix is returned.

    """

    X = cudata.layers[layer] if layer is not None else cudata.X

    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    computed_on = layer if layer else "cudata.X"
    settings_dict = dict(theta=theta, clip=clip, computed_on=computed_on)
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if clip is None:
        n = X.shape[0]
        clip = cp.sqrt(n, dtype=cp.float32)
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    theta = cp.array([1 / theta], dtype=cp.float32)
    clip = cp.array([clip], dtype=cp.float32)
    sums_cells = cp.zeros(X.shape[0], dtype=cp.float32)
    sums_genes = cp.zeros(X.shape[1], dtype=cp.float32)

    if cpx.scipy.sparse.issparse(X):
        residuals = cp.zeros(X.shape, dtype=cp.float32)
        if cpx.scipy.sparse.isspmatrix_csc(X):
            block = (8,)
            grid = (int(math.ceil(X.shape[1] / block[0])),)
            _sparse_kernel_sum_csc(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[1]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            _sparse_kernel_norm_res_csc(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        elif cpx.scipy.sparse.isspmatrix_csr(X):
            block = (8,)
            grid = (int(math.ceil(X.shape[0] / block[0])),)
            _sparse_kernel_sum_csr(
                grid,
                block,
                (X.indptr, X.indices, X.data, sums_genes, sums_cells, X.shape[0]),
            )
            sum_total = 1 / sums_genes.sum().squeeze()
            _sparse_kernel_norm_res_csr(
                grid,
                block,
                (
                    X.indptr,
                    X.indices,
                    X.data,
                    sums_cells,
                    sums_genes,
                    residuals,
                    sum_total,
                    clip,
                    theta,
                    X.shape[0],
                    X.shape[1],
                ),
            )
        else:
            raise ValueError(
                "Please transform you sparse matrix into CSR or CSC format."
            )
    else:
        residuals = cp.zeros(X.shape, dtype=cp.float32)
        block = (8, 8)
        grid = (
            math.ceil(residuals.shape[0] / block[0]),
            math.ceil(residuals.shape[1] / block[1]),
        )
        _dense_kernel_sum(
            grid,
            block,
            (X, sums_cells, sums_genes, residuals.shape[0], residuals.shape[1]),
        )
        sum_total = 1 / sums_genes.sum().squeeze()
        _kernel_norm_res_dense(
            grid,
            block,
            (
                X,
                residuals,
                sums_cells,
                sums_genes,
                sum_total,
                clip,
                theta,
                residuals.shape[0],
                residuals.shape[1],
            ),
        )

    if inplace == True:
        cudata.uns["pearson_residuals_normalization"] = settings_dict
        if layer:
            cudata.layers[layer] = residuals
        else:
            cudata.X = residuals
    else:
        return residuals
