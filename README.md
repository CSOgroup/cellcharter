<div align="center">
<img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/cellcharter.png" width="400px">

**A Python package for the identification, characterization and comparison of spatial clusters from spatial -omics data.**

---

<p align="center">
  <a href="https://cellcharter.readthedocs.io/en/latest/" target="_blank">Documentation</a> •
  <a href="https://cellcharter.readthedocs.io/en/latest/notebooks/codex_mouse_spleen.html" target="_blank">Examples</a> •
  <a href="https://www.biorxiv.org/content/10.1101/2023.01.10.523386v1" target="_blank">Paper</a>
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
  <img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/spatial_clusters.png" width="500px">
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
-   [Tutorials][link-tutorial]

## Installation

1. CellCharter uses Python < 3.11 and [PyTorch](https://pytorch.org) < 2.0.0. If you are planning to use a GPU, make sure to download and install the correct version of PyTorch first.
2. Install CellCharter using pip:

```bash
pip install cellcharter
```

3. In CellCharter, only the dimensionality reduction and batch correction step is dependent on the data type. In particular, it uses:

-   [scVI](https://github.com/scverse/scvi-tools) for spatial transcriptomics data such as 10x Visium and Xenium, Nanostring CosMx, Vizgen MERSCOPE, Stereo-seq, DBiT-seq, MERFISH and seqFISH data.
-   A modified version of [scArches](https://github.com/theislab/scarches)'s TRVAE model for spatial proteomics data such as Akoya CODEX, Lunaphore COMET, CyCIF, IMC and MIBI-TOF data.

We suggest using mamba to install the dependencies.
One example of installation that uses the GPU and is focused on spatial transcriptomics data is:

```bash
conda create -n cellcharter-env -c conda-forge python=3.10 mamba
conda activate cellcharter-env
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install cellcharter
pip install scvi-tools==1.0.3
```

## Contribution

If you found a bug or you want to propose a new feature, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/CSOgroup/cellcharter/issues
[link-docs]: https://cellcharter.readthedocs.io
[link-api]: https://cellcharter.readthedocs.io/en/latest/api.html
[link-tutorial]: https://cellcharter.readthedocs.io/en/latest/notebooks/codex_mouse_spleen.html
