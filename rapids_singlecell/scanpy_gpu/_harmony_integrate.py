from anndata import AnnData


def harmony_integrate(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    **kwargs,
):
    """\
    Use harmonypy to integrate different experiments.
    Harmony is an algorithm for integrating single-cell
    data from multiple experiments. This function uses the python
    gpu-computing based port of Harmony, to integrate single-cell data
    stored in an AnnData object. As Harmony works by adjusting the
    principal components, this function should be run after performing
    PCA but before computing the neighbor graph, as illustrated in the
    example below.
    Parameters
    ----------
    adata
        The annotated data matrix.
    key
        The name of the column in ``adata.obs`` that differentiates
        among experiments/batches.
    basis
        The name of the field in ``adata.obsm`` where the PCA table is
        stored. Defaults to ``'X_pca'``, which is the default for
        ``sc.tl.pca()``.
    adjusted_basis
        The name of the field in ``adata.obsm`` where the adjusted PCA
        table will be stored after running this function. Defaults to
        ``X_pca_harmony``.
    kwargs
        Any additional arguments will be passed to
        ``harmonypy.run_harmony()``.
    Returns
    -------
    Updates adata with the field ``adata.obsm[obsm_out_field]``,
    containing principal components adjusted by Harmony such that
    different experiments are integrated.
    """
    from . import _harmonpy_gpu
    harmony_out = _harmonpy_gpu.run_harmony(adata.obsm[basis], adata.obs, key, **kwargs)

    adata.obsm[adjusted_basis] = harmony_out.Z_corr.T.get()
