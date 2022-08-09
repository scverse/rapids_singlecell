#
# created by Severin Dicks (IBSM, Freiburg)
#
#

import cupy as cp
import cupyx as cpx
import anndata

import numpy as np
import pandas as pd
import scipy
import math
from scipy import sparse
from typing import Any, Union, Optional, Mapping

import warnings

from scipy.sparse import issparse as issparse_cpu
from cupyx.scipy.sparse import issparse as issparse_gpu

from cuml.linear_model import LinearRegression


class Layer_Mapping(dict):
    """
    Dictonary subclass for layers handeling in cunnData
    """
    def __init__(self, shape):
        super().__init__({})
        self.shape = shape
    
    def update_shape(self,shape):
        self.shape = shape

    def __setitem__(self, key, item):
        if self.shape == item.shape:
            super().__setitem__(key, item)
        else:
            raise ValueError(f"Shape of {key} does not match `.X`")

class cunnData:
    """
    The cunnData objects can be used as an AnnData replacement for the inital preprocessing of single cell Datasets. It replaces some of the most common preprocessing steps within scanpy for annData objects.
    It can be initalized with a preexisting annData object or with a countmatrix and seperate Dataframes for var and obs. Index of var will be used as gene_names. Initalization with an AnnData object is advised.
    """
    uns = {}
    def __init__(
        self,
        X: Optional[Union[np.ndarray,sparse.spmatrix, cp.array, cp.sparse.csr_matrix]] = None,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Mapping[str, Any]] = None,
        layers: Optional[Mapping[str, Any]] = None,
        adata: Optional[anndata.AnnData] = None):
            if adata:
                if not issparse_cpu(adata.X):
                    inter = scipy.sparse.csr_matrix(adata.X)
                    self.X = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                    del inter
                else:
                    self.X = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                self.obs = adata.obs.copy()
                self.var = adata.var.copy()
                self.uns = adata.uns.copy()
                self.layers = Layer_Mapping(self.shape)
                if adata.layers:
                    for key, matrix in adata.layers.items():
                        if not issparse_cpu(matrix):
                            inter = scipy.sparse.csr_matrix(adata.X)
                            inter = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                            
                        else:
                            inter = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                    self.layers[key] = inter.copy()
                    del inter
                
            else:
                if issparse_gpu(X):
                    self.X = X                
                elif not issparse_cpu(X):
                    inter = scipy.sparse.csr_matrix(X)
                    self.X = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                    del inter
                else:
                    self.X = cp.sparse.csr_matrix(X, dtype=cp.float32)

                self.obs = obs
                self.var = var
                self.uns = uns
                self.layers = Layer_Mapping(self.shape)
                if layers:
                    for key, matrix in layers.items():
                        if issparse_gpu(matrix):
                            inter = matrix.copy()               
                        elif not issparse_cpu(X):
                            inter = scipy.sparse.csr_matrix(adata.X)
                            inter = cp.sparse.csr_matrix(inter, dtype=cp.float32)
                        else:
                            inter = cp.sparse.csr_matrix(adata.X, dtype=cp.float32)
                        self.layers[key] = inter.copy()
                        del inter
    
    @property
    def shape(self):
        return self.X.shape
    @property
    def nnz(self):
        return self.X.nnz
    
    @property
    def obs_names(self):
        return  self.obs.index

    @property
    def var_names(self):
        return self.var.index

    def __getitem__(self, index):
        """
        Currently only works for `obs`
        """
        index = index.to_numpy()
        self.X = self.X[index,:]
        self.layers.update_shape(self.shape)
        if self.layers:
            for key, matrix in self.layers.items():
                self.layers[key] = matrix[index, :]
        return(cunnData(X = self.X,obs = self.obs.loc[index,:],var = self.var,uns=self.uns,layers= self.layers))


    def to_AnnData(self):
        """
        Takes the cunnData object and creates an AnnData object
        
        Returns
        -------
            annData object
        
        """
        adata = anndata.AnnData(self.X.get())
        adata.obs = self.obs.copy()
        adata.var = self.var.copy()
        adata.uns = self.uns.copy()
        if self.layers:
            for key, matrix in self.layers.items():
                adata.layers[key] = matrix.get()
        return adata
    
    def calc_gene_qc(self, batchsize = None):
        """
        Filters out genes that expressed in less than a specified number of cells

        Parameters
        ----------
        
            batchsize: int (default: None)
                Number of rows to be processed together This can be adjusted for performance to trade-off memory use.
            
        Returns
        -------
            updated `.var` with `n_cells` and `n_counts`
            filtered cunndata object inplace for genes less than the threshhold
        
        """
        if batchsize:
            n_batches = math.ceil(self.X.shape[0] / batchsize)
            n_counts = cp.zeros(shape=(n_batches,self.X.shape[1]))
            n_cells= cp.zeros(shape=(n_batches,self.X.shape[1]))
            for batch in range(n_batches):
                start_idx = batch * batchsize
                stop_idx = min(batch * batchsize + batchsize, self.X.shape[0])
                arr_batch = self.X[start_idx:stop_idx]
                arr_batch = arr_batch.tocsc()
                n_cells_batch = cp.diff(arr_batch.indptr).ravel()
                n_cells[batch,:]=n_cells_batch
                n_counts_batch = arr_batch.sum(axis = 0).ravel()
                n_counts[batch,:]=n_counts_batch
            self.var["n_cells"] = cp.asnumpy(n_cells.sum(axis= 0).ravel())
            self.var["n_counts"] = cp.asnumpy(n_counts.sum(axis= 0).ravel())
        else:
            self.X = self.X.tocsc()
            n_cells = cp.diff(self.X.indptr).ravel()
            self.X = self.X.tocsr()
            n_counts = self.X.sum(axis = 0).ravel()
            self.var["n_cells"] = cp.asnumpy(n_cells)
            self.var["n_counts"] = cp.asnumpy(n_counts)


    def filter_genes(self, qc_var = "n_cells", min_count = None, max_count = None, batchsize = None, verbose =True):
        """
        Filter genes that have greater than a max number of genes or less than
        a minimum number of a feature in a given `.var` columns. Can so far only be used for numerical columns.
        You can run this function on 'n_cells' or 'n_counts' with a previous columns in `.var`.
        
        Parameters
        ----------
        qc_var: str (default: n_cells)
            column in `.var` with numerical entries to filter against
            
        min_count : float
            Lower bound on number of a given feature to keep gene

        max_count : float
            Upper bound on number of a given feature to keep gene
        
        batchsize: int (default: None)
            only needed if you run `filter_genes` before `calculate_qc` or `calc_gene_qc` on 'n_genes' or 'n_counts'. Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        verbose: bool (default: True)
            Print number of discarded genes
        
        Returns
        -------
        a filtered cunnData object inplace
        
        """
        
        if qc_var in self.var.keys():
            if min_count is not None and max_count is not None:
                thr=np.where((self.var[qc_var] <= max_count) &  (min_count <= self.var[qc_var]))[0]
            elif min_count is not None:
                thr=np.where(self.var[qc_var] >= min_count)[0]
            elif max_count is not None:
                thr=np.where(self.var[qc_var] <= max_count)[0]

            if verbose:
                print(f"filtered out {self.var.shape[0]-thr.shape[0]} genes based on {qc_var}")
            self.X = self.X.tocsr()
            self.X = self.X[:, thr]
            self.X = self.X.tocsr()
            self.var = self.var.iloc[cp.asnumpy(thr)]
            self.layers.update_shape(self.shape)
            if self.layers:
                for key, matrix in self.layers.items():
                    self.layers[key] = matrix[:, thr]
            
        elif qc_var in ["n_cells","n_counts"]:
            self.calc_gene_qc(batchsize = batchsize)    
            if min_count is not None and max_count is not None:
                thr=np.where((self.var[qc_var] <= max_count) &  (min_count <= self.var[qc_var]))[0]
            elif min_count is not None:
                thr=np.where(self.var[qc_var] >= min_count)[0]
            elif max_count is not None:
                thr=np.where(self.var[qc_var] <= max_count)[0]

            if verbose:
                print(f"filtered out {self.var.shape[0]-thr.shape[0]} genes based on {qc_var}")
            self.X = self.X[:, thr]
            self.layers.update_shape(self.shape)
            if self.layers:
                for key, matrix in self.layers.items():
                    self.layers[key] = matrix[:, thr]
            self.var = self.var.iloc[cp.asnumpy(thr)]
        else:
            print(f"please check qc_var")


        
    def caluclate_qc(self, qc_vars = None, batchsize = None):
        """
        Calculates basic qc Parameters. Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
        Loosly based on calculate_qc_metrics from scanpy [Wolf et al. 2018]. Updates .obs with columns with qc data.
        
        Parameters
        ----------
        qc_vars: str, list (default: None)
            Keys for boolean columns of .var which identify variables you could want to control for (e.g. Mito). Run flag_gene_family first
            
        batchsize: int (default: None)
            Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        Returns
        -------
        adds the following columns in .obs
        n_counts
            number of counts per cell
        n_genes
            number of genes per cell
        for qc_var in qc_vars
            total_qc_var
                number of counts per qc_var (e.g total counts mitochondrial genes)
            percent_qc_vars
                
                Proportion of counts of qc_var (percent of counts mitochondrial genes)
        
        """      
        if batchsize:
            n_batches = math.ceil(self.X.shape[0] / batchsize)
            n_genes = []
            n_counts = []
            if "n_cells" not in self.var.keys() or  "n_counts" not in self.var.keys():
                self.calc_gene_qc(batchsize = batchsize)    
            if qc_vars:
                if type(qc_vars) is str:
                    qc_var_total = []
                    
                elif type(qc_vars) is list:
                    qc_var_total = []
                    for i in range(len(qc_vars)):
                        my_list = []
                        qc_var_total.append(my_list)
                        
            for batch in range(n_batches):
                batch_size = batchsize
                start_idx = batch * batch_size
                stop_idx = min(batch * batch_size + batch_size, self.X.shape[0])
                arr_batch = self.X[start_idx:stop_idx]
                n_genes.append(cp.diff(arr_batch.indptr).ravel().get())
                n_counts.append(arr_batch.sum(axis=1).ravel().get())
                if qc_vars:
                    if type(qc_vars) is str:
                        qc_var_total.append(arr_batch[:,self.var[qc_vars]].sum(axis=1).ravel().get())

                    elif type(qc_vars) is list:
                        for i in range(len(qc_vars)):
                             qc_var_total[i].append(arr_batch[:,self.var[qc_vars[i]]].sum(axis=1).ravel().get())
                        
                
            self.obs["n_genes"] = np.concatenate(n_genes)
            self.obs["n_counts"] = np.concatenate(n_counts)
            if qc_vars:
                if type(qc_vars) is str:
                    self.obs["total_"+qc_vars] = np.concatenate(qc_var_total)
                    self.obs["percent_"+qc_vars] =self.obs["total_"+qc_vars]/self.obs["n_counts"]*100
                elif type(qc_vars) is list:
                    for i in range(len(qc_vars)):
                        self.obs["total_"+qc_vars[i]] = np.concatenate(qc_var_total[i])
                        self.obs["percent_"+qc_vars[i]] =self.obs["total_"+qc_vars[i]]/self.obs["n_counts"]*100
        else:
            if "n_cells" not in self.var.keys() or  "n_counts" not in self.var.keys():
                self.calc_gene_qc(batchsize = None) 
            self.obs["n_genes"] = cp.asnumpy(cp.diff(self.X.indptr)).ravel()
            self.X = self.X.tocsc()
            self.obs["n_counts"] = cp.asnumpy(self.X.sum(axis=1)).ravel()

               
            if qc_vars:
                if type(qc_vars) is str:
                    self.obs["total_"+qc_vars]=cp.asnumpy(self.X[:,self.var[qc_vars]].sum(axis=1))
                    self.obs["percent_"+qc_vars]=self.obs["total_"+qc_vars]/self.obs["n_counts"]*100

                elif type(qc_vars) is list:
                    for qc_var in qc_vars:
                        self.obs["total_"+qc_var]=cp.asnumpy(self.X[:,self.var[qc_var]].sum(axis=1))
                        self.obs["percent_"+qc_var]=self.obs["total_"+qc_var]/self.obs["n_counts"]*100
            self.X = self.X.tocsr()

    def flag_gene_family(self, gene_family_name = str, gene_family_prefix = None, gene_list= None):
        """
        Flags a gene or gene_familiy in .var with boolean. (e.g all mitochondrial genes).
        Please only choose gene_family prefix or gene_list
        
        Parameters
        ----------
        gene_family_name: str
            name of colums in .var where you want to store informationa as a boolean
            
        gene_family_prefix: str
            prefix of the gene familiy (eg. mt- for all mitochondrial genes in mice)
            
        gene_list: list
            list of genes to flag in .var
        
        Returns
        -------
        adds the boolean column in .var 
        
        """
        if gene_family_prefix:
            self.var[gene_family_name] = cp.asnumpy(self.var.index.str.startswith(gene_family_prefix)).ravel()
        if gene_list:
            self.var[gene_family_name] = cp.asnumpy(self.var.index.isin(gene_list)).ravel()
    
    def filter_cells(self, qc_var, min_count=None, max_count=None, batchsize = None,verbose=True):
        """
        Filter cells that have greater than a max number of genes or less than
        a minimum number of a feature in a given .obs columns. Can so far only be used for numerical columns.
        It is recommended to run `calculated_qc` before using this function. You can run this function on n_genes or n_counts before running `calculated_qc`.
        
        Parameters
        ----------
        qc_var: str
            column in .obs with numerical entries to filter against
            
        min_count : float
            Lower bound on number of a given feature to keep cell

        max_count : float
            Upper bound on number of a given feature to keep cell
        
        batchsize: int (default: None)
            only needed if you run `filter_cells` before `calculate_qc` on 'n_genes' or 'n_counts'. Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        verbose: bool (default: True)
            Print number of discarded cells
        
        Returns
        -------
        a filtered cunnData object inplace
        
        """
        if qc_var in self.obs.keys(): 
            inter = np.array
            if min_count is not None and max_count is not None:
                inter=np.where((self.obs[qc_var] < max_count) &  (min_count< self.obs[qc_var]))[0]
            elif min_count is not None:
                inter=np.where(self.obs[qc_var] > min_count)[0]
            elif max_count is not None:
                inter=np.where(self.obs[qc_var] < max_count)[0]
            else:
                print(f"Please specify a cutoff to filter against")
            if verbose:
                print(f"filtered out {self.obs.shape[0]-inter.shape[0]} cells")
            self.X = self.X[inter,:]
            self.obs = self.obs.iloc[inter]
            self.layers.update_shape(self.shape)
            if self.layers:
                for key, matrix in self.layers.items():
                    self.layers[key] = matrix[inter,:]
        elif qc_var in ['n_genes','n_counts']:
            print(f"Running calculate_qc for 'n_genes' or 'n_counts'")
            self.caluclate_qc(batchsize=batchsize)
            inter = np.array
            if min_count is not None and max_count is not None:
                inter=np.where((self.obs[qc_var] < max_count) &  (min_count< self.obs[qc_var]))[0]
            elif min_count is not None:
                inter=np.where(self.obs[qc_var] > min_count)[0]
            elif max_count is not None:
                inter=np.where(self.obs[qc_var] < max_count)[0]
            else:
                print(f"Please specify a cutoff to filter against")
            if verbose:
                print(f"filtered out {self.obs.shape[0]-inter.shape[0]} cells")
            self.X = self.X[inter,:]
            self.obs = self.obs.iloc[inter]
            self.layers.update_shape(self.shape)
            if self.layers:
                for key, matrix in self.layers.items():
                    self.layers[key] = matrix[inter,:]
        else:
            print(f"Please check qc_var.")
            

        
    def normalize_total(self, target_sum):
        """
        Normalizes rows in matrix so they sum to `target_sum`

        Parameters
        ----------

        target_sum : int
            Each row will be normalized to sum to this value
        
        
        Returns
        -------
        
        a normalized sparse Matrix to a specified target sum
        
        """
        csr_arr = self.X
        mul_kernel = cp.RawKernel(r'''
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
            ''', 'mul_kernel')

        mul_kernel((math.ceil(csr_arr.shape[0] / 32.0),), (32,),
                       (csr_arr.indptr,
                        csr_arr.data,
                        csr_arr.shape[0],
                       int(target_sum)))

        self.X = csr_arr
    
    def log1p(self):
        """
        Calculated the natural logarithm of one plus the sparse marttix, element-wise inlpace in cunnData object.
        """
        self.X = self.X.log1p()
        self.uns["log1p"] = {"base": None}

    def normalize_pearson_residuals(self,
        theta: float = 100,
        clip: Optional[float] = None,
        check_values: bool = True,
        layer: Optional[str] = None,
        inplace = True):

        X = self.layers[layer] if layer is not None else self.X
        X = X.copy()
        if check_values and not _check_nonnegative_integers(X):
            warnings.warn(
                "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
                UserWarning,
        )
        if theta <= 0:
            raise ValueError('Pearson residuals require theta > 0')
        if clip is None:
            n = X.shape[0]
            clip = cp.sqrt(n)
        if clip < 0:
            raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

        if type(X) is not cpx.scipy.sparse.csc.csc_matrix:
            sums_cells = X.sum(axis=1)
            X =X.tocsr()
            sums_genes = X.sum(axis=0)
        elif type(X) is not cpx.scipy.sparse.csr.csr_matrix:
            sums_genes = X.sum(axis=0)
            X =X.tocsc()
            sums_cells = X.sum(axis=1)
        
        sum_total = sums_genes.sum().squeeze()
        mu = sums_cells @ sums_genes / sum_total
        X = X - mu
        X = X / cp.sqrt( mu + mu**2 / theta)
        X = cp.clip(X, a_min=-clip, a_max=clip)
        if inplace == True:
            if layer:
                self.layers[layer]= X
            else:
                self.X= X
        else:
            return X

    
    def highly_varible_genes(
        self,
        layer = None,
        min_mean = 0.0125,
        max_mean =3,
        min_disp= 0.5,
        max_disp =np.inf,
        n_top_genes = None,
        flavor = 'seurat',
        n_bins = 20,
        span = 0.3,
        check_values: bool = True,
        theta:int = 100,
        clip = None,
        chunksize:int = 1000,
        n_samples:int = 10000, 
        batch_key = None):
        """
        Annotate highly variable genes. 
        Expects logarithmized data, except when `flavor='seurat_v3','pearson_residuals','poisson_gene_selection'`, in which count data is expected.
        
        Reimplentation of scanpy's function. 
        Depending on flavor, this reproduces the R-implementations of Seurat, Cell Ranger, Seurat v3 and Pearson Residuals.
        Flavor `poisson_gene_selection` is an implementation of scvi, which is based on M3Drop. It requiers gpu accelerated pytorch to be installed.
        
        For these dispersion-based methods, the normalized dispersion is obtained by scaling 
        with the mean and standard deviation of the dispersions for genes falling into a given 
        bin for mean expression of genes. This means that for each bin of mean expression, 
        highly variable genes are selected.
        
        For Seurat v3, a normalized variance for each gene is computed. First, the data
        are standardized (i.e., z-score normalization per feature) with a regularized
        standard deviation. Next, the normalized variance is computed as the variance
        of each gene after the transformation. Genes are ranked by the normalized variance.


        Parameters
        ----------
        layer
            If provided, use `cudata.layers[layer]` for expression values instead of `cudata.X`.
        min_mean: float (default: 0.0125)
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_mean: float (default: 3)
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        min_disp: float (default: 0.5)
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_disp: float (default: inf)
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        n_top_genes: int (defualt: None)
            Number of highly-variable genes to keep.
        flavor : {`seurat`, `cell_ranger`, `seurat_v3`, `pearson_residuals`, `poisson_gene_selection`} (default: 'seurat')
            Choose the flavor for identifying highly variable genes. For the dispersion based methods in their default workflows, Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.
        n_bins : int (default: 20)
            Number of bins for binning the mean gene expression. Normalization is done with respect to each bin. If just a single gene falls into a bin, the normalized dispersion is artificially set to 1.
        span : float (default: 0.3)
            The fraction of the data (cells) used when estimating the variance in the loess
            model fit if `flavor='seurat_v3'`.
        check_values: bool (default: True)
            Check if counts in selected layer are integers. A Warning is returned if set to True.
            Only used if `flavor='seurat_v3'` or `'pearson_residuals'`.
        theta: int (default: 1000)
            The negative binomial overdispersion parameter `theta` for Pearson residuals.
             Higher values correspond to less overdispersion (`var = mean + mean^2/theta`), and `theta=np.Inf` corresponds to a Poisson model.
        clip: float (default: None)
            Only used if `flavor='pearson_residuals'`.
            Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval `[-sqrt(n_obs), sqrt(n_obs)]`, where `n_obs` is the number of cells in the dataset (default behavior).
                * If any scalar `c`, residuals are clipped to the interval `[-c, c]`. Set `clip=np.Inf` for no clipping.
        chunksize: int (default: 1000)
            If `flavor='pearson_residuals'` or `'poisson_gene_selection'`, this dertermines how many genes are processed at
            once while computing the residual variance. Choosing a smaller value will reduce
            the required memory.
        n_samples: int (default: 10000)
            The number of Binomial samples to use to estimate posterior probability
            of enrichment of zeros for each gene (only for `flavor='poisson_gene_selection'`).
        batch_key:
            If specified, highly-variable genes are selected within each batch separately and merged.
            
        Returns
        -------
        
        upates .var with the following fields
        highly_variable : bool
            boolean indicator of highly-variable genes
        means
            means per gene
        dispersions
            For dispersion-based flavors, dispersions per gene
        dispersions_norm
            For dispersion-based flavors, normalized dispersions per gene
        variances
            For `flavor='seurat_v3','pearson_residuals'`, variance per gene
        variances_norm
            For `flavor='seurat_v3'`, normalized variance per gene, averaged in
            the case of multiple batches
        residual_variances : float
            For `flavor='pearson_residuals'`, residual variance per gene. Averaged in the
            case of multiple batches.
        highly_variable_rank : float
            For `flavor='seurat_v3','pearson_residuals'`, rank of the gene according to normalized
            variance, median rank in the case of multiple batches
        highly_variable_nbatches : int
            If batch_key is given, this denotes in how many batches genes are detected as HVG
        highly_variable_intersection : bool
            If batch_key is given, this denotes the genes that are highly variable in all batches     
        """
        if flavor == 'seurat_v3':
            _highly_variable_genes_seurat_v3(
                cudata = self,
                layer=layer,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                span=span,
                check_values = check_values,
            )
        elif flavor == 'pearson_residuals':
            _highly_variable_pearson_residuals(
                cudata = self,
                theta= theta,
                clip = clip,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                check_values = check_values,
                layer=layer,
                chunksize= chunksize)
        elif flavor == 'poisson_gene_selection':
            _poisson_gene_selection(
                cudata =self,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                check_values = check_values,
                layer=layer,
                n_samples = n_samples,
                minibatch_size= chunksize)
        else:
            if batch_key is None:
                X = self.layers[layer] if layer is not None else self.X
                df = _highly_variable_genes_single_batch(
                    X.tocsc(),
                    min_disp=min_disp,
                    max_disp=max_disp,
                    min_mean=min_mean,
                    max_mean=max_mean,
                    n_top_genes=n_top_genes,
                    n_bins=n_bins,
                    flavor=flavor)
            else:
                self.obs[batch_key] = self.obs[batch_key].astype("category")
                batches = self.obs[batch_key].cat.categories
                df = []
                genes = self.var.index.to_numpy()
                for batch in batches:
                    inter_matrix = self.X[np.where(self.obs[batch_key]==batch)[0],].tocsc()
                    thr_org = cp.diff(inter_matrix.indptr).ravel()
                    thr = cp.where(thr_org >= 1)[0]
                    thr_2 = cp.where(thr_org < 1)[0]
                    inter_matrix = inter_matrix[:, thr]
                    thr = thr.get()
                    thr_2 = thr_2.get()
                    inter_genes = genes[thr]
                    other_gens_inter = genes[thr_2]
                    hvg_inter = _highly_variable_genes_single_batch(inter_matrix,
                                                                    min_disp=min_disp,
                                                                    max_disp=max_disp,
                                                                    min_mean=min_mean,
                                                                    max_mean=max_mean,
                                                                    n_top_genes=n_top_genes,
                                                                    n_bins=n_bins,
                                                                    flavor=flavor)
                    hvg_inter["gene"] = inter_genes
                    missing_hvg = pd.DataFrame(
                        np.zeros((len(other_gens_inter), len(hvg_inter.columns))),
                        columns=hvg_inter.columns,
                    )
                    missing_hvg['highly_variable'] = missing_hvg['highly_variable'].astype(bool)
                    missing_hvg['gene'] = other_gens_inter
                    hvg = hvg_inter.append(missing_hvg, ignore_index=True)
                    idxs = np.concatenate((thr, thr_2))
                    hvg = hvg.loc[np.argsort(idxs)]
                    df.append(hvg)
                
                df = pd.concat(df, axis=0)
                df['highly_variable'] = df['highly_variable'].astype(int)
                df = df.groupby('gene').agg(
                    dict(
                        means=np.nanmean,
                        dispersions=np.nanmean,
                        dispersions_norm=np.nanmean,
                        highly_variable=np.nansum,
                    )
                )
                df.rename(
                    columns=dict(highly_variable='highly_variable_nbatches'), inplace=True
                )
                df['highly_variable_intersection'] = df['highly_variable_nbatches'] == len(
                    batches
                )
                if n_top_genes is not None:
                    # sort genes by how often they selected as hvg within each batch and
                    # break ties with normalized dispersion across batches
                    df.sort_values(
                        ['highly_variable_nbatches', 'dispersions_norm'],
                        ascending=False,
                        na_position='last',
                        inplace=True,
                    )
                    df['highly_variable'] = False
                    df.highly_variable.iloc[:n_top_genes] = True
                    df = df.loc[genes]
                else:
                    df = df.loc[genes]
                    dispersion_norm = df.dispersions_norm.values
                    dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
                    gene_subset = np.logical_and.reduce(
                        (
                            df.means > min_mean,
                            df.means < max_mean,
                            df.dispersions_norm > min_disp,
                            df.dispersions_norm < max_disp,
                        )
                    )
                    df['highly_variable'] = gene_subset
            
            self.var["highly_variable"] =df['highly_variable'].values
            self.var["means"] = df['means'].values
            self.var["dispersions"]=df['dispersions'].values
            self.var["dispersions_norm"]=df['dispersions_norm'].values
            self.uns['hvg'] = {'flavor': flavor}
            if batch_key is not None:
                self.var['highly_variable_nbatches'] = df[
                    'highly_variable_nbatches'
                ].values
                self.var['highly_variable_intersection'] = df[
                    'highly_variable_intersection'
                ].values
    

        
    def filter_highly_variable(self):
        """
        Filters the cunndata object for highly_variable genes. Run highly_varible_genes first.
        
        Returns
        -------
        
        updates cunndata object to only contain highly variable genes.
        
        """
        if "highly_variable" in self.var.keys():
            thr = np.where(self.var["highly_variable"] == True)[0]
            self.X =self.X.tocsc()
            self.X = self.X[:, thr]
            self.var = self.var.iloc[cp.asnumpy(thr)]
            self.layers.update_shape(self.shape)
            if self.layers:
                for key, matrix in self.layers.items():
                    self.layers[key] = matrix[:,thr]
        else:
            print(f"Please calculate highly variable genes first")
            
    def regress_out(self, keys, verbose=False):

        """
        Use linear regression to adjust for the effects of unwanted noise
        and variation. 
        Parameters
        ----------

        adata
            The annotated data matrix.
        keys
            Keys for numerical observation annotation on which to regress on.

        verbose : bool
            Print debugging information

        Returns
        -------
        updates cunndata object with the corrected data matrix


        """
        
        if type(self.X) is not cpx.scipy.sparse.csc.csc_matrix:
            self.X = self.X.tocsc()

        dim_regressor= 2
        if type(keys)is list:
            dim_regressor = len(keys)+1

        regressors = cp.ones((self.X.shape[0]*dim_regressor)).reshape((self.X.shape[0], dim_regressor), order="F")
        if dim_regressor==2:
            regressors[:, 1] = cp.array(self.obs[keys]).ravel()
        else:
            for i in range(dim_regressor-1):
                regressors[:, i+1] = cp.array(self.obs[keys[i]]).ravel()

        outputs = cp.empty(self.X.shape, dtype=self.X.dtype, order="F")

        if self.X.shape[0] < 100000 and cpx.scipy.sparse.issparse(self.X):
            self.X = self.X.todense()
        
        for i in range(self.X.shape[1]):
            if verbose and i % 500 == 0:
                print("Regressed %s out of %s" %(i, self.X.shape[1]))
            X = regressors
            y = self.X[:,i]
            outputs[:, i] = _regress_out_chunk(X, y)
        self.X = outputs
    
    
    def scale(self, max_value=10):
        """
        Scales matrix to unit variance and clips values
        Parameters
        ----------
        max_value : int
                    After scaling matrix to unit variance,
                    values will be clipped to this number
                    of std deviations.
        Return
        ------
        updates cunndata object with a scaled cunndata.X
        """
        if type(self.X) is not cp._core.core.ndarray:
            print("densifying _.X")
            X = self.X.toarray()
        else:
            X =self.X
        mean = X.mean(axis=0)
        X -= mean
        del mean
        stddev = cp.sqrt(X.var(axis=0))
        X /= stddev
        del stddev
        self.X = cp.clip(X,a_max=max_value)
        
def _regress_out_chunk(X, y):
    """
    Performs a data_cunk.shape[1] number of local linear regressions,
    replacing the data in the original chunk w/ the regressed result.

    Parameters
    ----------

    X : cupy.ndarray of shape (n_cells, 3)
        Matrix of regressors

    y : cupy.sparse.spmatrix of shape (n_cells,)
        Sparse matrix containing a single column of the cellxgene matrix

    Returns
    -------

    dense_mat : cupy.ndarray of shape (n_cells,)
        Adjusted column
    """
    if cp.sparse.issparse(y):
        y = y.todense()

    lr = LinearRegression(fit_intercept=False, output_type="cupy")
    lr.fit(X, y, convert_dtype=True)
    return y.reshape(y.shape[0],) - lr.predict(X).reshape(y.shape[0])

def _highly_variable_genes_single_batch(X,min_mean = 0.0125,max_mean =3,min_disp= 0.5,max_disp =np.inf, n_top_genes = None, flavor = 'seurat', n_bins = 20):
        """\
        See `highly_variable_genes`.
        Returns
        -------
        A DataFrame that contains the columns
        `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
        """
        if flavor == 'seurat':
            X = X.expm1()
        mean, var = _get_mean_var(X)
        mean[mean == 0] = 1e-12
        disp = var/mean
        if flavor == 'seurat':  # logarithmized mean as in Seurat
            disp[disp == 0] = np.nan
            disp = cp.log(disp)
            mean = cp.log1p(mean)
        df = pd.DataFrame()
        mean = mean.get()
        disp = disp.get()
        df['means'] = mean
        df['dispersions'] = disp
        if flavor == 'seurat':
            df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
            disp_grouped = df.groupby('mean_bin')['dispersions']
            disp_mean_bin = disp_grouped.mean()
            disp_std_bin = disp_grouped.std(ddof=1)
            # retrieve those genes that have nan std, these are the ones where
            # only a single gene fell in the bin and implicitly set them to have
            # a normalized disperion of 1
            one_gene_per_bin = disp_std_bin.isnull()
            gen_indices = np.where(one_gene_per_bin[df['mean_bin'].values])[0].tolist()

            # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
            # but there’s still a dtype error without “.value”.
            disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[
                one_gene_per_bin.values
            ].values
            disp_mean_bin[one_gene_per_bin.values] = 0
            # actually do the normalization
            df['dispersions_norm'] = (
                df['dispersions'].values  # use values here as index differs
                - disp_mean_bin[df['mean_bin'].values].values
            ) / disp_std_bin[df['mean_bin'].values].values

        elif flavor == 'cell_ranger':
            from statsmodels import robust
            df['mean_bin'] = pd.cut(
                    df['means'],
                    np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 5)), np.inf],
                )
            disp_grouped = df.groupby('mean_bin')['dispersions']
            disp_median_bin = disp_grouped.median()
            with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    disp_mad_bin = disp_grouped.apply(robust.mad)
                    df['dispersions_norm'] = (
                        df['dispersions'].values - disp_median_bin[df['mean_bin'].values].values
                    ) / disp_mad_bin[df['mean_bin'].values].values

        dispersion_norm = df['dispersions_norm'].values
        if n_top_genes is not None:
            dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
            dispersion_norm[::-1].sort()# interestingly, np.argpartition is slightly slower
            if n_top_genes > X.shape[1]:
                n_top_genes = X.shape[1]
            disp_cut_off = dispersion_norm[n_top_genes - 1]
            gene_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
        else:
            dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
            gene_subset = np.logical_and.reduce(
                (
                    mean > min_mean,
                    mean < max_mean,
                    dispersion_norm > min_disp,
                    dispersion_norm < max_disp,
                )
            )
        df['highly_variable'] = gene_subset
        return df

def _highly_variable_genes_seurat_v3(
    cudata: cunnData,
    layer: Optional[str] = None,
    n_top_genes: int = None,
    batch_key: Optional[str] = None,
    span: float = 0.3,
    check_values = True):
    """\
    See `highly_variable_genes`.
    For further implementation details see https://www.overleaf.com/read/ckptrbgzzzpg
    Returns
    -------
    updates `.var` with the following fields:
    highly_variable : bool
        boolean indicator of highly-variable genes.
    **means**
        means per gene.
    **variances**
        variance per gene.
    **variances_norm**
        normalized variance per gene, averaged in the case of multiple batches.
    highly_variable_rank : float
        Rank of the gene according to normalized variance, median rank in the case of multiple batches.
    highly_variable_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as HVG.
    """
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='seurat_v3'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    try:
        from skmisc.loess import loess
    except ImportError:
        raise ImportError(
            'Please install skmisc package via `pip install --user scikit-misc'
        )

    df = pd.DataFrame(index=cudata.var.index)
    X = cudata.layers[layer] if layer is not None else cudata.X
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
            UserWarning,
        )

    mean, var = _get_mean_var(X.tocsc())
    df['means'], df['variances'] = mean.get(), var.get()
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(cudata.shape[0], dtype=int))
    else:
        batch_info = cudata.obs[batch_key].values

    norm_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = X[batch_info == b]
        mean, var = _get_mean_var(X_batch.tocsc())
        not_const = var > 0
        estimat_var = cp.zeros(X_batch.shape[1], dtype=np.float64)

        y = cp.log10(var[not_const])
        x = cp.log10(mean[not_const])
        model = loess(x.get(), y.get(), span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = cp.sqrt(10**estimat_var)
        batch_counts = X_batch
        N = X_batch.shape[0]
        vmax = cp.sqrt(N)
        clip_val = reg_std * vmax + mean
        mask = batch_counts.data > clip_val[batch_counts.indices]
        batch_counts.data[mask] = clip_val[batch_counts.indices[mask]]
        squared_batch_counts_sum = cp.array(batch_counts.power(2).sum(axis=0))
        batch_counts_sum = cp.array(batch_counts.sum(axis=0))

        norm_gene_var = (1 / ((N - 1) * cp.square(reg_std))) * (
            (N * cp.square(mean))
            + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )

        norm_gene_vars.append(norm_gene_var.reshape(1, -1))
    norm_gene_vars = cp.concatenate(norm_gene_vars, axis=0)
    ranked_norm_gene_vars = cp.argsort(cp.argsort(-norm_gene_vars, axis=1), axis=1)

    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = cp.sum(
        (ranked_norm_gene_vars < n_top_genes).astype(int), axis=0
    )
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top_genes] = np.nan
    ranked_norm_gene_vars = ranked_norm_gene_vars.get()
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
    df['highly_variable_nbatches'] = num_batches_high_var.get()
    df['highly_variable_rank'] = median_ranked
    df['variances_norm'] = cp.mean(norm_gene_vars, axis=0).get()
    sorted_index = (
        df[['highly_variable_rank', 'highly_variable_nbatches']]
        .sort_values(
            ['highly_variable_rank', 'highly_variable_nbatches'],
            ascending=[True, False],
            na_position='last',
        )
        .index
    )
    df['highly_variable'] = False
    df.loc[sorted_index[: int(n_top_genes)], 'highly_variable'] = True
    cudata.var['highly_variable'] = df['highly_variable'].values
    cudata.var['highly_variable_rank'] = df['highly_variable_rank'].values
    cudata.var['means'] = df['means'].values
    cudata.var['variances'] = df['variances'].values
    cudata.var['variances_norm'] = df['variances_norm'].values.astype(
            'float64', copy=False
        )
    cudata.var['highly_variable_nbatches'] = df[
                'highly_variable_nbatches'
            ].values
    cudata.uns['hvg'] = {'flavor': 'seurat_v3'}

def _highly_variable_pearson_residuals(cudata: cunnData,
    theta: float = 100,
    clip: Optional[float] = None,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    check_values: bool = True,
    layer: Optional[str] = None,
    chunksize= 1000):
    """
    Select highly variable genes using analytic Pearson residuals.
    Pearson residuals of a negative binomial offset model are computed
    (with overdispersion `theta` shared across genes). By default, overdispersion
    `theta=100` is used and residuals are clipped to `sqrt(n_obs)`. Finally, genes
    are ranked by residual variance.
    Expects raw count input.
    """
    X = cudata.layers[layer] if layer is not None else cudata.X
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='pearson_residuals'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    if theta <= 0:
        raise ValueError('Pearson residuals require theta > 0')
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(cudata.shape[0], dtype=int))
    else:
        batch_info = cudata.obs[batch_key].values
        
    n_batches = len(np.unique(batch_info))
    residual_gene_vars = []
    for b in np.unique(batch_info):
        X_batch = X[batch_info == b].tocsc()
        thr_org = cp.diff(X_batch.indptr).ravel()
        nonzero_genes = cp.array(thr_org >= 1)
        X_batch = X_batch[:, nonzero_genes]
        if clip is None:
            n = X_batch.shape[0]
            clip = cp.sqrt(n)
        if clip < 0:
            raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
        sums_cells = X_batch.sum(axis=1)
        X_batch =X_batch.tocsr()
        sums_genes = X_batch.sum(axis=0)
        sum_total = sums_genes.sum().squeeze()
        # Compute pearson residuals in chunks
        residual_gene_var = cp.empty((X_batch.shape[1]))
        X_batch = X_batch.tocsc()
        for start in np.arange(0, X_batch.shape[1], chunksize):
            stop = start + chunksize
            mu = cp.array(sums_cells @ sums_genes[:, start:stop] / sum_total)
            X_dense = X_batch[:, start:stop].toarray()
            residuals = (X_dense - mu) / cp.sqrt(mu + mu**2 / theta)
            residuals = cp.clip(residuals, a_min=-clip, a_max=clip)
            residual_gene_var[start:stop] = cp.var(residuals, axis=0)

        unmasked_residual_gene_var = cp.zeros(len(nonzero_genes))
        unmasked_residual_gene_var[nonzero_genes] = residual_gene_var
        residual_gene_vars.append(unmasked_residual_gene_var.reshape(1, -1))
        
    residual_gene_vars = cp.concatenate(residual_gene_vars, axis=0)
    # Get rank per gene within each batch
    # argsort twice gives ranks, small rank means most variable
    ranks_residual_var = cp.argsort(cp.argsort(-residual_gene_vars, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(np.float32)
    # count in how many batches a genes was among the n_top_genes
    highly_variable_nbatches = cp.sum(
        (ranks_residual_var < n_top_genes).astype(int), axis=0
    ).get()
    ranks_residual_var[ranks_residual_var >= n_top_genes] = np.nan
    ranks_residual_var= ranks_residual_var.get()
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    # Median rank across batches, ignoring batches in which gene was not selected
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)
    means, variances = _get_mean_var(X.tocsc())
    means, variances = means.get(), variances.get()
    df = pd.DataFrame.from_dict(
        dict(
            means=means,
            variances=variances,
            residual_variances=cp.mean(residual_gene_vars, axis=0).get(),
            highly_variable_rank=medianrank_residual_var,
            highly_variable_nbatches=highly_variable_nbatches.astype(np.int64),
            highly_variable_intersection=highly_variable_nbatches == n_batches,
        )
    )
    df = df.set_index(cudata.var_names)
    df.sort_values(
        ['highly_variable_nbatches', 'highly_variable_rank'],
        ascending=[False, True],
        na_position='last',
        inplace=True,
    )
    high_var = np.zeros(df.shape[0], dtype=bool)
    high_var[:n_top_genes] = True
    df['highly_variable'] = high_var
    df = df.loc[cudata.var_names, :]
    
    computed_on = layer if layer else 'adata.X'
    cudata.uns['hvg'] = {'flavor': 'pearson_residuals', 'computed_on': computed_on}
    cudata.var['means'] = df['means'].values
    cudata.var['variances'] = df['variances'].values
    cudata.var['residual_variances'] = df['residual_variances']
    cudata.var['highly_variable_rank'] = df['highly_variable_rank'].values
    if batch_key is not None:
        cudata.var['highly_variable_nbatches'] = df[
            'highly_variable_nbatches'
        ].values
        cudata.var['highly_variable_intersection'] = df[
            'highly_variable_intersection'
        ].values
    cudata.var['highly_variable'] = df['highly_variable'].values

def _poisson_gene_selection(
    cudata:cunnData,
    layer: Optional[str] = None,
    n_top_genes: int = None,
    n_samples: int = 10000,
    batch_key: str = None,
    minibatch_size: int = 1000,
    check_values:bool = True,
    **kwargs,
) -> None:
    """
    Rank and select genes based on the enrichment of zero counts.
    Enrichment is considered by comparing data to a Poisson count model.
    This is based on M3Drop: https://github.com/tallulandrews/M3Drop
    The method accounts for library size internally, a raw count matrix should be provided.
    Instead of Z-test, enrichment of zeros is quantified by posterior
    probabilites from a binomial model, computed through sampling.
    Parameters
    ----------
    cudata
        cunnData object (with sparse X matrix).
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        How many variable genes to select.
    n_samples
        The number of Binomial samples to use to estimate posterior probability
        of enrichment of zeros for each gene.
    batch_key
        key in adata.obs that contains batch info. If None, do not use batch info.
        Defatult: ``None``.
    minibatch_size
        Size of temporary matrix for incremental calculation. Larger is faster but
        requires more RAM or GPU memory. (The default should be fine unless
        there are hundreds of millions cells or millions of genes.)
    Returns
    -------
    Depending on `inplace` returns calculated metrics (:class:`~pd.DataFrame`) or
    updates `.var` with the following fields
    highly_variable : bool
        boolean indicator of highly-variable genes
    **observed_fraction_zeros**
        fraction of observed zeros per gene
    **expected_fraction_zeros**
        expected fraction of observed zeros per gene
    prob_zero_enrichment : float
        Probability of zero enrichment, median across batches in the case of multiple batches
    prob_zero_enrichment_rank : float
        Rank of the gene according to probability of zero enrichment, median rank in the case of multiple batches
    prob_zero_enriched_nbatches : int
        If batch_key is given, this denotes in how many batches genes are detected as zero enriched
    """
    
    try:
        import torch
    except ImportError:
        raise ImportError(
            'Please install pytorch package via `pip install pytorch'
        )
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='seurat_v3'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    
    X = cudata.layers[layer] if layer is not None else cudata.X
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(cudata.shape[0], dtype=int))
    else:
        batch_info = cudata.obs[batch_key].values

    prob_zero_enrichments = []
    obs_frac_zeross = []
    exp_frac_zeross = []
    
    with torch.no_grad():
        for b in np.unique(batch_info):
            X_batch = X[batch_info == b]
            total_counts = torch.tensor(X_batch.sum(1).ravel(), device = "cuda")
            X_batch = X_batch.tocsc()
            # Calculate empirical statistics.
            sum_0 = X_batch.sum(axis =0).ravel()
            scaled_means = torch.tensor(sum_0 / sum_0.sum(), device = "cuda")

            observed_fraction_zeros = torch.tensor(
                cp.asarray(1.0 - cp.diff(X_batch.indptr).ravel() / X_batch.shape[0]).ravel(),
                device = "cuda")
            # Calculate probability of zero for a Poisson model.
            # Perform in batches to save memory.
            minibatch_size = min(total_counts.shape[0], minibatch_size)
            n_batches = total_counts.shape[0] // minibatch_size

            expected_fraction_zeros = torch.zeros(scaled_means.shape, device="cuda")

            for i in range(n_batches):
                total_counts_batch = total_counts[
                    i * minibatch_size : (i + 1) * minibatch_size
                ]
                # Use einsum for outer product.
                expected_fraction_zeros += torch.exp(
                    -torch.einsum("i,j->ij", [scaled_means, total_counts_batch])
                ).sum(1)

            total_counts_batch = total_counts[(i + 1) * minibatch_size :]
            expected_fraction_zeros += torch.exp(
                -torch.einsum("i,j->ij", [scaled_means, total_counts_batch])
            ).sum(1)
            expected_fraction_zeros /= X_batch.shape[0]

            # Compute probability of enriched zeros through sampling from Binomial distributions.
            observed_zero = torch.distributions.Binomial(probs=observed_fraction_zeros)
            expected_zero = torch.distributions.Binomial(probs=expected_fraction_zeros)

            #extra_zeros = torch.zeros(expected_fraction_zeros.shape, device="cuda")
            
            
            extra_zeros = observed_zero.sample((n_samples,))>expected_zero.sample((n_samples,))
            #for i in range(n_samples):
            #    extra_zeros += observed_zero.sample() > expected_zero.sample()
            
            extra_zeros = extra_zeros.sum(0)
            prob_zero_enrichment = (extra_zeros / n_samples).cpu().numpy()

            obs_frac_zeros = observed_fraction_zeros.cpu().numpy()
            exp_frac_zeros = expected_fraction_zeros.cpu().numpy()

            # Clean up memory (tensors seem to stay in GPU unless actively deleted).
            del scaled_means
            del total_counts
            del expected_fraction_zeros
            del observed_fraction_zeros
            del extra_zeros
            torch.cuda.empty_cache()

            prob_zero_enrichments.append(prob_zero_enrichment.reshape(1, -1))
            obs_frac_zeross.append(obs_frac_zeros.reshape(1, -1))
            exp_frac_zeross.append(exp_frac_zeros.reshape(1, -1))

    # Combine per batch results
    prob_zero_enrichments = np.concatenate(prob_zero_enrichments, axis=0)
    obs_frac_zeross = np.concatenate(obs_frac_zeross, axis=0)
    exp_frac_zeross = np.concatenate(exp_frac_zeross, axis=0)

    ranked_prob_zero_enrichments = prob_zero_enrichments.argsort(axis=1).argsort(axis=1)
    median_prob_zero_enrichments = np.median(prob_zero_enrichments, axis=0)

    median_obs_frac_zeross = np.median(obs_frac_zeross, axis=0)
    median_exp_frac_zeross = np.median(exp_frac_zeross, axis=0)

    median_ranked = np.median(ranked_prob_zero_enrichments, axis=0)

    num_batches_zero_enriched = np.sum(
        ranked_prob_zero_enrichments >= (cudata.shape[1] - n_top_genes), axis=0
    )

    df = pd.DataFrame(index=np.array(cudata.var_names))
    df["observed_fraction_zeros"] = median_obs_frac_zeross
    df["expected_fraction_zeros"] = median_exp_frac_zeross
    df["prob_zero_enriched_nbatches"] = num_batches_zero_enriched
    df["prob_zero_enrichment"] = median_prob_zero_enrichments
    df["prob_zero_enrichment_rank"] = median_ranked

    df["highly_variable"] = False
    sort_columns = ["prob_zero_enriched_nbatches", "prob_zero_enrichment_rank"]
    top_genes = df.nlargest(n_top_genes, sort_columns).index
    df.loc[top_genes, "highly_variable"] = True

    cudata.uns["hvg"] = {"flavor": "poisson_zeros"}
    cudata.var["highly_variable"] = df["highly_variable"].values
    cudata.var["observed_fraction_zeros"] = df["observed_fraction_zeros"].values
    cudata.var["expected_fraction_zeros"] = df["expected_fraction_zeros"].values
    cudata.var["prob_zero_enriched_nbatches"] = df[
        "prob_zero_enriched_nbatches"
    ].values
    cudata.var["prob_zero_enrichment"] = df["prob_zero_enrichment"].values
    cudata.var["prob_zero_enrichment_rank"] = df["prob_zero_enrichment_rank"].values

    if batch_key is not None:
        cudata.var["prob_zero_enriched_nbatches"] = df["prob_zero_enriched_nbatches"].values



def _get_mean_var(X):
    mean = (X.sum(axis =0)/X.shape[0]).ravel()
    X.data **= 2
    inter = (X.sum(axis =0)/X.shape[0]).ravel()
    var = inter - mean ** 2
    return mean, var

def _check_nonnegative_integers(X):
    """Checks values of X to ensure it is count data"""
    data = X.data
    # Check no negatives
    if cp.signbit(data).any():
        return False
    elif cp.any(~cp.equal(cp.mod(data, 1), 0)):
        return False
    else:
        return True