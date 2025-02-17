<div align="center">
<img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/cellcharter.png" width="400px">

**A Python package for the identification, characterization and comparison of spatial clusters from spatial omics data.**

---

<p align="center">
  <a href="https://cellcharter.readthedocs.io/en/latest/" target="_blank">Documentation</a> •
  <a href="https://cellcharter.readthedocs.io/en/latest/notebooks/codex_mouse_spleen.html" target="_blank">Examples</a> •
  <a href="https://doi.org/10.1038/s41588-023-01588-4" target="_blank">Paper</a> •
  <a href="https://www.biorxiv.org/content/10.1101/2023.01.10.523386v2" target="_blank">Preprint</a>
</p>

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/CSOgroup/cellcharter/test.yaml?branch=main
[link-tests]: https://github.com/CSOgroup/cellcharter/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/cellcharter

</div>

## Background

<p>
  Spatial clustering (or spatial domain identification) determines cellular niches characterized by specific admixing of these populations. It assigns cells to clusters based on both their intrinsic features (e.g., protein or mRNA expression), and the features of neighboring cells in the tissue.
</p>
<p align="center">
  <img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/spatial_clusters.png" width="500px">
</p>

<p>
CellCharter is able to automatically identify spatial domains and offers a suite of approaches for cluster characterization and comparison.
</p>
<p align="center">
  <img src="https://github.com/CSOgroup/cellcharter/raw/main/docs/_static/cellcharter_workflow.png" width="800px">
</p>

## Features

- **Identify niches for multiple samples**: By combining the power of scVI and scArches, CellCharter can identify domains for multiple samples simultaneously, even with in presence of batch effects.
- **Scalability**: CellCharter can handle large datasets with millions of cells and thousands of features. The possibility to run it on GPUs makes it even faster
- **Flexibility**: CellCharter can be used with different types of spatial omics data, such as spatial transcriptomics, proteomics, epigenomics and multiomics data. The only difference is the method used for dimensionality reduction and batch effect removal.
    - Spatial transcriptomics: CellCharter has been tested on [scVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCVI) with Zero-inflated negative binomial distribution.
    - Spatial proteomics: CellCharter has been tested on a version of [scArches](https://docs.scarches.org/en/latest/api/models.html#scarches.models.TRVAE), modified to use Mean Squared Error loss instead of the default Negative Binomial loss.
    - Spatial epigenomics: CellCharter has been tested on [scVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html#scvi.model.SCVI) with Poisson distribution.
    - Spatial multiomics: it's possible to use multi-omics models such as [MultiVI](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.MULTIVI.html#scvi.model.MULTIVI), or use the concatenation of the results from the different models.
- **Best candidates for the number of domains**: CellCharter offers a [method to find multiple best candidates](https://cellcharter.readthedocs.io/en/latest/generated/cellcharter.tl.ClusterAutoK.html) for the number of domains, based on the stability of a certain number of domains across multiple runs.
- **Domain characterization**: CellCharter provides a set of tools to characterize and compare the spatial domains, such as domain proportion, cell type enrichment, (differential) neighborhood enrichment, and domain shape characterization.

Since CellCharter 0.3.0, we moved the implementation of the Gaussian Mixture Model (GMM) from [PyCave](https://github.com/borchero/pycave), not maintained anymore, to [TorchGMM](https://github.com/CSOgroup/torchgmm), a fork of PyCave maintained by the CSOgroup. This change allows us to have a more stable and maintained implementation of GMM that is compatible with the most recent versions of PyTorch.

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api].
- [Tutorials][link-tutorial]

## Installation

1. Create a conda or pyenv environment
2. Install Python >= 3.8 and [PyTorch](https://pytorch.org) >= 1.12.0. If you are planning to use a GPU, make sure to download and install the correct version of PyTorch first from [here](https://pytorch.org/get-started/locally/).
3. Install the library used for dimensionality reduction and batch effect removal according to the data type you are planning to analyze:
    - [scVI](https://github.com/scverse/scvi-tools) for spatial transcriptomics and/or epigenomics data such as 10x Visium and Xenium, Nanostring CosMx, Vizgen MERSCOPE, Stereo-seq, DBiT-seq, MERFISH and seqFISH data.
    - A modified version of [scArches](https://github.com/theislab/scarches)'s TRVAE model for spatial proteomics data such as Akoya CODEX, Lunaphore COMET, CyCIF, IMC and MIBI-TOF data.
4. Install CellCharter using pip:

```bash
pip install cellcharter
```

We suggest using `mamba` to install the dependencies.
Installing the latest version of the dependencies (in particular `scvi-tools` and `spatialdata`) may lead to dependency conflicts.
However, this should not be a problem because CellCharter doesn't use any of the mismatching features.

We report here an example of an installation aimed at analyzing spatial transcriptomics data (and thus installing `scvi-tools`).
This example is based on a Linux CentOS 7 system with an NVIDIA A100 GPU.

```bash
conda create -n cellcharter-env -c conda-forge python mamba
conda activate cellcharter-env
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install scvi-tools
pip install cellcharter
```

Note: a different system may require different commands to install PyTorch and JAX. Refer to their respective documentation for more details.

## Contribution

If you found a bug or you want to propose a new feature, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/CSOgroup/cellcharter/issues
[link-docs]: https://cellcharter.readthedocs.io
[link-api]: https://cellcharter.readthedocs.io/en/latest/api.html
[link-tutorial]: https://cellcharter.readthedocs.io/en/latest/notebooks/codex_mouse_spleen.html
