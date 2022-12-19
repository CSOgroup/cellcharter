<div align="center">
<img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/cellcharter.png" width="400px">

**A Python package for the identification, characterization and comparison of spatial clusters from spatial -omics data.**

---

<p align="center">
  <a href="https://cellcharter.readthedocs.io/en/latest/">Documentation</a> •
  <a href="https://cellcharter.readthedocs.io/en/latest/notebooks/codex_mouse_spleen.html">Examples</a> •
  Paper
</p>

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/CSOgroup/cellcharter/test.yaml?branch=main
[link-tests]: https://github.com/CSOgroup/cellcharter/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/cellcharter

</div>

## Background

<p>
  Spatial clustering determines cellular niches characterized by specific admixing of these populations. It assigns cells to clusters based on both their intrinsic features (e.g., protein or mRNA expression), and the features of neighboring cells in the tissue.
</p>
<p align="center">
  <img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/spatial_clusters.png" width="400px">
</p>

<p>
CellCharter is able to automatically identify spatial clusters, and offers a suite of approaches for cluster characterization and comparison.
</p>
<p align="center">
  <img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/cellcharter_workflow.png" width="800px">
</p>

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

CellCharter uses [PyTorch](https://pytorch.org). If you are planning to use a GPu, make sure to download and install the correct version of PyTorch first.

Then you are ready to install CellCharter.

```bash
pip install git+https://github.com/CSOgroup/cellcharter.git@main
```

## Contribution

If you found a bug or you wnat to propose a new feature, please use the [issue tracker][issue-tracker].

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/CSOgroup/cellcharter/issues
[changelog]: https://cellcharter.readthedocs.io/latest/changelog.html
[link-docs]: https://cellcharter.readthedocs.io
[link-api]: https://cellcharter.readthedocs.io/latest/api.html
