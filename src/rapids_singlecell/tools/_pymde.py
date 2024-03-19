from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def mde(
    adata: AnnData,
    *,
    device: Literal["cpu", "cuda"] | None = None,
    n_neighbors: int = 15,
    n_pcs: int = None,
    use_rep: str = None,
    **kwargs,
) -> None:
    """
    Util to run :func:`pymde.preserve_neighbors` for visualization of single cell embeddings.

    Parameters
    ----------
        adata
            Annotated data matrix.
        device
            Whether to run on cpu or gpu ("cuda"). If None, tries to run on gpu if available.
        n_neighbors
            use this many neighbors
        n_pcs
            use this many PCs
        use_rep
            use this obsm keys (defaults to `X_pca`)
        kwargs
            Keyword args to :func:`pymde.preserve_neighbors`

    Returns
    -------
        Updates `adata` with the following fields.

            **X_mde** : `np.ndarray` (`adata.obs`, dtype `float`)
                X_mde coordinates of data.

    Notes
    -----
        This function adapted from scvi-tools.
        The appropriateness of use of visualization of high-dimensional spaces in single-
        cell omics remains an open research questions. See:
        Chari, Tara, Joeyta Banerjee, and Lior Pachter. "The specious art of single-cell genomics." bioRxiv (2021).
        If you use this function in your research please cite:
        Agrawal, Akshay, Alnur Ali, and Stephen Boyd. "Minimum-distortion embedding." arXiv preprint arXiv:2103.02559 (2021).
    """
    try:
        import pymde
        import torch
    except ImportError:
        raise ImportError("Please install pymde package via `pip install pymde`")

    if use_rep is None:
        data = adata.obsm["X_pca"]
    else:
        data = adata.obsm[use_rep]

    if isinstance(data, pd.DataFrame):
        data = data.values
    if n_pcs is not None:
        data = data[:, :n_pcs]

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    _kwargs = {
        "embedding_dim": 2,
        "constraint": pymde.Standardized(),
        "repulsive_fraction": 0.7,
        "verbose": False,
        "device": device,
        "n_neighbors": n_neighbors,
    }
    _kwargs.update(kwargs)

    emb = pymde.preserve_neighbors(data, **_kwargs).embed(verbose=_kwargs["verbose"])

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
        torch.cuda.empty_cache()

    adata.obsm["X_mde"] = emb
