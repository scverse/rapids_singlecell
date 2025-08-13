[![Stars](https://img.shields.io/github/stars/scverse/rapids_singlecell?style=flat&logo=GitHub&color=blue)](https://github.com/scverse/rapids_singlecell/stargazers)
[![PyPI](https://img.shields.io/pypi/v/rapids-singlecell?logo=PyPI)](https://pypi.org/project/rapids-singlecell)
[![Downloads](https://static.pepy.tech/badge/rapids-singlecell)](https://pepy.tech/project/rapids-singlecell)
[![Documentation Status](https://readthedocs.org/projects/rapids-singlecell/badge/?version=latest)](https://rapids-singlecell.readthedocs.io/en/latest/?badge=latest)
[![Build and Test](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml/badge.svg)](https://github.com/scverse/rapids_singlecell/actions/workflows/test-gpu.yml)
[![Chat](https://img.shields.io/badge/zulip-join_chat-%2367b08f.svg)](https://scverse.zulipchat.com)

# rapids-singlecell: GPU-Accelerated Single-Cell Analysis within scverse®

rapids-singlecell provides GPU-accelerated single-cell analysis with an AnnData-first API. It is largely compatible with Scanpy and includes selected functionality from Squidpy and decoupler. Computations use CuPy and NVIDIA RAPIDS for performance on large datasets.

- **GPU acceleration**: Common single-cell workflows on `AnnData` run on the GPU.
- **Ecosystem compatibility**: Works with Scanpy APIs; includes pieces from Squidpy and decoupler.
- **Simple installation**: Available via Conda and PyPI.

## Documentation

For more information please have a look through the [documentation](https://rapids-singlecell.readthedocs.io/en/latest/)


## Citation

If you use this code, please cite: [![DOI](https://zenodo.org/badge/364573913.svg)](https://zenodo.org/badge/latestdoi/364573913)

Please also consider citing: [rapids-single-cell-examples](https://zenodo.org/badge/latestdoi/265649968) and  [scanpy](https://doi.org/10.1186/s13059-017-1382-0)

In addition to that please cite the methods' original research articles in the [scanpy documentation](https://scanpy.readthedocs.io/en/latest/references.html)

If you use the accelerated decoupler functions please cite [decoupler](https://doi.org/10.1093/bioadv/vbac016)

[//]: # (numfocus-fiscal-sponsor-attribution)

rapids-singlecell is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
