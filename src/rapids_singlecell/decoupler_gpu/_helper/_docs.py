from __future__ import annotations

from docrep import DocstringProcessor

_plot = """\
ax
    An existing :class:`matplotlib.axes._axes.Axes` instance to plot on. If ``None``,
    a new figure and axes will be created.
figsize
    Size of the figure in inches as (width, height).
dpi
    Dots per inch for the figure resolution.
return_fig
    If ``True``, plotting methods should return the figure object instead of showing it.
save
    If set, path to save the plot automatically to a file.

Returns
-------
If ``return_fig=True``, returns :class:`matplotlib.figure.Figure` instance.
"""

_features = """\
features
    Column names of ``mat``."""

_net = """\
net
    Dataframe in long format. Must include ``source`` and ``target`` columns, and optionally a ``weight`` column."""

_tmin = """\
tmin
    Minimum number of targets per source. Sources with fewer targets will be removed."""

_bsize = """\
bsize
    For large datasets in sparse format, this parameter controls how many observations are processed at once.
    Increasing this value speeds up computation but uses more memory."""

_verbose = """\
verbose
    Whether to display progress messages and additional execution details."""

_data = """\
data
    AnnData instance, DataFrame or tuple of [matrix, samples, features]."""

_layer = """\
layer
    Layer key name of an ``anndata.AnnData`` instance."""

_raw = """\
raw
    Whether to use the ``.raw`` attribute of ``anndata.AnnData``."""

_empty = """\
empty
    Whether to remove empty observations (rows) or features (columns)."""

_adata = """\
adata
    Annotated data matrix with observations (rows) and features (columns)."""

_times = """\
times
    Number of random permutations to do."""

_seed = """\
seed
    Random seed to use."""

_inplace = """\
inplace
    Whether to perform the operation in the same object."""

_min_cells = """\
min_cells
    Minimum number of cells per sample."""

_min_counts = """\
min_counts
    Minimum number of counts per sample."""

_key = """\
key
    ``adata.obsm`` key to use."""

_cmap = """\
cmap
    Colormap to use."""

_min_prop_prop = """\
min_prop
    Minimum proportion of cells that express a gene in a sample."""

_min_smpls = """\
min_smpls
    Minimum number of samples with bigger or equal proportion of cells with expression than ``min_prop``."""

_group = """\
group
    Name of the ``adata.obs`` column to group by. If None, it assumes that all samples belong to one group."""

_lib_size = """\
lib_size
    Library size. If None, default to the sum of reads per sample."""

_min_count = """\
min_count
    Minimum count required per gene for at least some samples."""

_min_total_count = """\
min_total_count
    Minimum total count required per gene across all samples."""

_large_n = """\
large_n
    Number of samples per group that is considered to be "large"."""

_min_prop_expr = """\
min_prop
    Minimum proportion of samples in the smallest group that express the gene."""

_organism = """\
organism
    The organism of interest. By default human."""

_license = """\
license
    Which license to use, available options are: academic, commercial, or nonprofit.
    By default, is set to academic to retrieve all possible interactions.
    Users are expected to comply with license regulations according to their affiliation."""

_df = """\
df
    Result of ``decoupler.bm.benchmark``."""

_hue = """\
hue
    Grouping variable that will produce different colors."""

_palette = """\
palette
    Method for choosing the colors to use"""

_y = """\
y
    Grouping variable to plot on y axis."""

_order = """\
order
    The name of the column in ``adata.obs`` to consider for ordering."""

_label = """\
label
    The name of the column in ``adata.obs`` to consider for coloring the grouping. By default ``None``."""

_nbins = """\
nbins
    Number of bins to use."""

_vmin = """\
vmin
    The value representing the lower limit of the color scale."""

_vcenter = """\
vcenter
    The value representing the center of the color scale."""

_vmax = """\
vmax
    The value representing the upper limit of the color scale."""

_data_plot = """\
data
    DataFrame containing feature-level statistics. Feature names must be in ``df.index``."""

_top = """\
top
    Number of top sources to plot."""

_yestest = """\
Finally, the obtained :math:`p_{value}` are adjusted by Benjamini-Hochberg correction."""

_notest = """\
This method does not perform statistical testing on :math:`ES` and therefore does not return :math:`p_{value}`."""

_returns = """\
Returns
-------

Enrichment scores :math:`ES` and, if applicable, adjusted :math:`p_{value}` by Benjamini-Hochberg.
"""

_tval = """\
tval
    Whether to return the t-value (``tval=True``) the coefficient of the fitted model (``tval=False``)."""

_params = f"""\
Parameters
----------
{_data}
{_net}
{_tmin}
{_layer}
{_raw}
{_empty}
{_bsize}
{_verbose}"""

docs = DocstringProcessor(
    plot=_plot,
    net=_net,
    tmin=_tmin,
    bsize=_bsize,
    verbose=_verbose,
    features=_features,
    data=_data,
    layer=_layer,
    raw=_raw,
    empty=_empty,
    adata=_adata,
    times=_times,
    seed=_seed,
    inplace=_inplace,
    min_cells=_min_cells,
    min_counts=_min_counts,
    key=_key,
    cmap=_cmap,
    min_prop_prop=_min_prop_prop,
    min_smpls=_min_smpls,
    group=_group,
    lib_size=_lib_size,
    min_count=_min_count,
    min_total_count=_min_total_count,
    large_n=_large_n,
    min_prop_expr=_min_prop_expr,
    organism=_organism,
    license=_license,
    df=_df,
    hue=_hue,
    palette=_palette,
    y=_y,
    order=_order,
    label=_label,
    nbins=_nbins,
    vmin=_vmin,
    vcenter=_vcenter,
    vmax=_vmax,
    data_plot=_data_plot,
    top=_top,
    params=_params,
    yestest=_yestest,
    notest=_notest,
    returns=_returns,
    tval=_tval,
)
