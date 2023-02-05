import cupy as cp
import numpy as np
import pandas as pd
import math
from ..cunnData import cunnData
from typing import Union

def calc_gene_qc(cudata:cunnData, batchsize:int = None)->None:

    """
    Filters out genes that expressed in less than a specified number of cells

    Parameters
    ----------
        cudata
            cunnData object
        batchsize
            Number of rows to be processed together This can be adjusted for performance to trade-off memory use.
            
    Returns
    -------
        updated `.var` with `n_cells` and `n_counts`
            filtered cunndata object inplace for genes less than the threshhold
    
    """
    if batchsize:
        n_batches = math.ceil(cudata.X.shape[0] / batchsize)
        n_counts = cp.zeros(shape=(n_batches,cudata.X.shape[1]))
        n_cells= cp.zeros(shape=(n_batches,cudata.X.shape[1]))
        for batch in range(n_batches):
            start_idx = batch * batchsize
            stop_idx = min(batch * batchsize + batchsize, cudata.X.shape[0])
            arr_batch = cudata.X[start_idx:stop_idx]
            arr_batch = arr_batch.tocsc()
            n_cells_batch = cp.diff(arr_batch.indptr).ravel()
            n_cells[batch,:]=n_cells_batch
            n_counts_batch = arr_batch.sum(axis = 0).ravel()
            n_counts[batch,:]=n_counts_batch
        cudata.var["n_cells"] = cp.asnumpy(n_cells.sum(axis= 0).ravel())
        cudata.var["n_counts"] = cp.asnumpy(n_counts.sum(axis= 0).ravel())
    else:
        cudata.X = cudata.X.tocsc()
        n_cells = cp.diff(cudata.X.indptr).ravel()
        cudata.X = cudata.X.tocsr()
        n_counts = cudata.X.sum(axis = 0).ravel()
        cudata.var["n_cells"] = cp.asnumpy(n_cells)
        cudata.var["n_counts"] = cp.asnumpy(n_counts)


def filter_genes(cudata:cunnData, 
                qc_var:str = "n_cells", 
                min_count:int = None,
                max_count:int = None,
                batchsize:int = None,
                verbose:bool =True)-> None:

    """
    Filter genes based on number of cells or counts.

    Filters genes, that have greater than a max number of genes or less than
    a minimum number of a feature in a given :attr:`.var` columns. Can so far only be used for numerical columns.
    You can run this function on 'n_cells' or 'n_counts' with a previous columns in :attr:`.var`.
    
    Parameters
    ----------
        cudata: 
            cunnData object

        qc_var
            column in :attr:`.var` with numerical entries to filter against
            
        min_count
            Lower bound on number of a given feature to keep gene

        max_count
            Upper bound on number of a given feature to keep gene
        
        batchsize
            only needed if you run `filter_genes` before `calculate_qc` or `calc_gene_qc` on 'n_genes' or 'n_counts'. Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
            
        verbose
            Print number of discarded genes
    
    Returns
    -------
        a filtered :class:`~rapids_singlecell.cunnData.cunnData` object inplace
    
    """
    
    if qc_var in cudata.var.keys():
        if min_count is not None and max_count is not None:
            thr=np.where((cudata.var[qc_var] <= max_count) &  (min_count <= cudata.var[qc_var]))[0]
        elif min_count is not None:
            thr=np.where(cudata.var[qc_var] >= min_count)[0]
        elif max_count is not None:
            thr=np.where(cudata.var[qc_var] <= max_count)[0]

        if verbose:
            print(f"filtered out {cudata.var.shape[0]-thr.shape[0]} genes based on {qc_var}")
        cudata.X = cudata.X.tocsr()
        cudata.X = cudata.X[:, thr]
        cudata.X = cudata.X.tocsr()
        cudata.var = cudata.var.iloc[cp.asnumpy(thr)]
        if cudata.layers:
            for key, matrix in cudata.layers.items():
                cudata.layers[key] = matrix[:, thr]
        if cudata.varm:
            for key, matrix in cudata.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    cudata.varm[key] = matrix.iloc[thr, :]
                else:
                    cudata.varm[key] = matrix[thr, :]
            
        
    elif qc_var in ["n_cells","n_counts"]:
        calc_gene_qc(cudata=cudata,batchsize = batchsize)    
        if min_count is not None and max_count is not None:
            thr=np.where((cudata.var[qc_var] <= max_count) &  (min_count <= cudata.var[qc_var]))[0]
        elif min_count is not None:
            thr=np.where(cudata.var[qc_var] >= min_count)[0]
        elif max_count is not None:
            thr=np.where(cudata.var[qc_var] <= max_count)[0]

        if verbose:
            print(f"filtered out {cudata.var.shape[0]-thr.shape[0]} genes based on {qc_var}")
        cudata.X = cudata.X[:, thr]
        if cudata.layers:
            for key, matrix in cudata.layers.items():
                cudata.layers[key] = matrix[:, thr]
        if cudata.varm:
            for key, matrix in cudata.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    cudata.varm[key] = matrix.iloc[thr, :]
                else:
                    cudata.varm[key] = matrix[thr, :]
            
        cudata.var = cudata.var.iloc[cp.asnumpy(thr)]
    else:
        print(f"please check qc_var")


        
def calculate_qc(cudata:cunnData, 
                qc_vars:Union[str, list] = None,
                batchsize:int = None)->None:

    """\
    Calculates basic qc Parameters. Calculates number of genes per cell (n_genes) and number of counts per cell (n_counts).
    Loosly based on calculate_qc_metrics from scanpy [Wolf et al. 2018]. Updates .obs with columns with qc data.
    
    Parameters
    ----------
        cudata: 
            cunnData object
        qc_vars
            Keys for boolean columns of :attr:`.var` which identify variables you could want to control for (e.g. Mito).
            Run flag_gene_family first
        batchsize
            Number of rows to be processed together. 
            This can be adjusted for performance to trade-off memory use.
            
    Returns
    -------
        adds the following columns in :attr:`.obs` :

            `n_counts`
                number of counts per cell
            `n_genes`
                number of genes per cell
            for `qc_var` in `qc_vars`
                `total_qc_var`
                    number of counts per qc_var (e.g total counts mitochondrial genes)
                `percent_qc_vars`
                    Proportion of counts of qc_var (percent of counts mitochondrial genes)
        
    """      
    if batchsize:
        n_batches = math.ceil(cudata.X.shape[0] / batchsize)
        n_genes = []
        n_counts = []
        if "n_cells" not in cudata.var.keys() or  "n_counts" not in cudata.var.keys():
            calc_gene_qc(cudata= cudata,batchsize = batchsize)    
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
            stop_idx = min(batch * batch_size + batch_size, cudata.X.shape[0])
            arr_batch = cudata.X[start_idx:stop_idx]
            n_genes.append(cp.diff(arr_batch.indptr).ravel().get())
            n_counts.append(arr_batch.sum(axis=1).ravel().get())
            if qc_vars:
                if type(qc_vars) is str:
                    qc_var_total.append(arr_batch[:,cudata.var[qc_vars]].sum(axis=1).ravel().get())

                elif type(qc_vars) is list:
                    for i in range(len(qc_vars)):
                            qc_var_total[i].append(arr_batch[:,cudata.var[qc_vars[i]]].sum(axis=1).ravel().get())
                    
            
        cudata.obs["n_genes"] = np.concatenate(n_genes)
        cudata.obs["n_counts"] = np.concatenate(n_counts)
        if qc_vars:
            if type(qc_vars) is str:
                cudata.obs["total_"+qc_vars] = np.concatenate(qc_var_total)
                cudata.obs["percent_"+qc_vars] =cudata.obs["total_"+qc_vars]/cudata.obs["n_counts"]*100
            elif type(qc_vars) is list:
                for i in range(len(qc_vars)):
                    cudata.obs["total_"+qc_vars[i]] = np.concatenate(qc_var_total[i])
                    cudata.obs["percent_"+qc_vars[i]] =cudata.obs["total_"+qc_vars[i]]/cudata.obs["n_counts"]*100
    else:
        if "n_cells" not in cudata.var.keys() or  "n_counts" not in cudata.var.keys():
            calc_gene_qc(cudata,batchsize = None) 
        cudata.obs["n_genes"] = cp.asnumpy(cp.diff(cudata.X.indptr)).ravel()
        cudata.X = cudata.X.tocsc()
        cudata.obs["n_counts"] = cp.asnumpy(cudata.X.sum(axis=1)).ravel()

            
        if qc_vars:
            if type(qc_vars) is str:
                cudata.obs["total_"+qc_vars]=cp.asnumpy(cudata.X[:,cudata.var[qc_vars]].sum(axis=1))
                cudata.obs["percent_"+qc_vars]=cudata.obs["total_"+qc_vars]/cudata.obs["n_counts"]*100

            elif type(qc_vars) is list:
                for qc_var in qc_vars:
                    cudata.obs["total_"+qc_var]=cp.asnumpy(cudata.X[:,cudata.var[qc_var]].sum(axis=1))
                    cudata.obs["percent_"+qc_var]=cudata.obs["total_"+qc_var]/cudata.obs["n_counts"]*100
        cudata.X = cudata.X.tocsr()

def flag_gene_family(cudata:cunnData,
                    gene_family_name:str,
                    gene_family_prefix:str = None,
                    gene_list:list= None)-> None:

    """
    Flags a gene or gene_familiy in .var with boolean. (e.g all mitochondrial genes).
    Please only choose gene_family prefix or gene_list
    
    Parameters
    ----------
        cudata
            cunnData object

        gene_family_name
            name of colums in .var where you want to store informationa as a boolean
            
        gene_family_prefix
            prefix of the gene familiy (eg. mt- for all mitochondrial genes in mice)
            
        gene_list
            list of genes to flag in `.var`
    
    Returns
    -------
        adds the boolean column in `.var` 
    
    """
    if gene_family_prefix:
        cudata.var[gene_family_name] = cp.asnumpy(cudata.var.index.str.startswith(gene_family_prefix)).ravel()
    if gene_list:
        cudata.var[gene_family_name] = cp.asnumpy(cudata.var.index.isin(gene_list)).ravel()
    
def filter_cells(cudata:cunnData, 
                qc_var:str,
                min_count:float=None,
                max_count:float=None,
                batchsize:bool = None,
                verbose:bool=True)->None:

    """\
    Filter cell outliers based on counts and numbers of genes expressed.

    Filter cells based on numerical columns in the :attr:`.obs` by selecting those with a feature count greater than a specified maximum or less than a specified minimum.
    It is recommended to run :func:`calculate_qc` before using this function. You can run this function on n_genes or n_counts before running :func:`calculate_qc`.
    
    Parameters
    ----------
        cudata: 
            cunnData object
        qc_var
            column in .obs with numerical entries to filter against
        min_count
            Lower bound on number of a given feature to keep cell
        max_count
            Upper bound on number of a given feature to keep cell
        batchsize
            only needed if you run `filter_cells` before `calculate_qc` on 'n_genes' or 'n_counts'. 
            Number of rows to be processed together. This can be adjusted for performance to trade-off memory use.
        verbose
            Print number of discarded cells
    
    Returns
    -------
       a filtered :class:`~rapids_singlecell.cunnData.cunnData` object inplace

    """
    if qc_var in cudata.obs.keys(): 
        inter = np.array
        if min_count is not None and max_count is not None:
            inter=np.where((cudata.obs[qc_var] < max_count) &  (min_count< cudata.obs[qc_var]))[0]
        elif min_count is not None:
            inter=np.where(cudata.obs[qc_var] > min_count)[0]
        elif max_count is not None:
            inter=np.where(cudata.obs[qc_var] < max_count)[0]
        else:
            print(f"Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {cudata.obs.shape[0]-inter.shape[0]} cells")
        cudata.X = cudata.X[inter,:]
        cudata.obs = cudata.obs.iloc[inter]
        cudata._update_shape()
        if cudata.layers:
            for key, matrix in cudata.layers.items():
                cudata.layers[key] = matrix[inter,:]
        if cudata.obsm:
            for key, matrix in cudata.obsm.items():
                cudata.obsm[key] = matrix[inter,:]
    elif qc_var in ['n_genes','n_counts']:
        print(f"Running calculate_qc for 'n_genes' or 'n_counts'")
        calculate_qc(cudata,batchsize=batchsize)
        inter = np.array
        if min_count is not None and max_count is not None:
            inter=np.where((cudata.obs[qc_var] < max_count) &  (min_count< cudata.obs[qc_var]))[0]
        elif min_count is not None:
            inter=np.where(cudata.obs[qc_var] > min_count)[0]
        elif max_count is not None:
            inter=np.where(cudata.obs[qc_var] < max_count)[0]
        else:
            print(f"Please specify a cutoff to filter against")
        if verbose:
            print(f"filtered out {cudata.obs.shape[0]-inter.shape[0]} cells")
        cudata.X = cudata.X[inter,:]
        cudata.obs = cudata.obs.iloc[inter]
        if cudata.layers:
            for key, matrix in cudata.layers.items():
                cudata.layers[key] = matrix[inter,:]
        if cudata.obsm:
            for key, matrix in cudata.obsm.items():
                cudata.obsm[key] = matrix[inter,:]
    else:
        print(f"Please check qc_var.")


def filter_highly_variable(cudata:cunnData)-> None:

    """
    Filters the :class:`~rapids_singlecell.cunnData.cunnData` object for highly_variable genes. Run highly_varible_genes first.
    
    Returns
    -------
        updates :class:`~rapids_singlecell.cunnData.cunnData` object to only contain highly variable genes.
    
    """
    if "highly_variable" in cudata.var.keys():
        thr = np.where(cudata.var["highly_variable"] == True)[0]
        cudata.X = cudata.X[:, thr]
        cudata.var = cudata.var.iloc[thr]
        if cudata.layers:
            for key, matrix in cudata.layers.items():
                cudata.layers[key] = matrix[:,thr]
        if cudata.varm:
            for key, matrix in cudata.varm.items():
                if isinstance(matrix, pd.DataFrame):
                    cudata.varm[key] = matrix.iloc[thr, :]
                else:
                    cudata.varm[key] = matrix[thr, :]
    else:
        print(f"Please calculate highly variable genes first")
