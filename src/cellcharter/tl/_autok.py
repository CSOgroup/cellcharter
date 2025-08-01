from __future__ import annotations

import concurrent.futures
import inspect
import json
import logging
import os
import pickle
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from torchgmm.base.utils.path import PathType
from tqdm.auto import tqdm

import cellcharter as cc

logger = logging.getLogger(__name__)


class ClusterAutoK:
    """
    Identify the best candidates for the number of clusters.

    Parameters
    ----------
    n_clusters
        Range for number of clusters (bounds included).
    max_runs
        Maximum number of repetitions for each value of number of clusters.
    convergence_tol
        Convergence tolerance for the clustering stability. If the Mean Absolute Percentage Error between consecutive iterations is below `convergence_tol` the algorithm stops without reaching `max_runs`.
    model_class
        Class of the model to be used for clustering. It must accept as `random_state` and `n_clusters` as initialization parameters.
    model_params
        Keyword args for `model_class`
    similarity_function
        The similarity function used between clustering results. Defaults to :func:`sklearn.metrics.fowlkes_mallows_score`.
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True)
    >>> cc.gr.remove_long_links(adata)
    >>> cc.gr.aggregate_neighbors(adata, n_layers=3)
    >>> model_params = {
            'random_state': 42,
            'trainer_params': {
                'accelerator':'cpu',
                'enable_progress_bar': False
            },
        }
    >>> models = cc.tl.ClusterAutoK(n_clusters=(2,10), model_class=cc.tl.GaussianMixture, model_params=model_params, max_runs=5)
    """

    #: The cluster assignments for each repetition and number of clusters.
    labels: dict

    #: The stability values of all combinations of runs between K and K-1, and between K and K+1
    stability: np.ndarray

    def __init__(
        self,
        n_clusters: tuple[int, int] | list[int],
        max_runs: int = 10,
        convergence_tol: float = 1e-2,
        model_class: type = None,
        model_params: dict = None,
        similarity_function: callable = None,
    ):
        self.n_clusters = (
            list(range(*(max(1, n_clusters[0] - 1), n_clusters[1] + 2)))
            if isinstance(n_clusters, tuple)
            else n_clusters
        )
        self.max_runs = max_runs
        self.convergence_tol = convergence_tol
        self.model_class = model_class if model_class else cc.tl.GaussianMixture
        self.model_params = model_params if model_params else {}
        self.similarity_function = similarity_function if similarity_function else fowlkes_mallows_score
        self.stability = []

    def fit(self, adata: ad.AnnData, use_rep: str = "X_cellcharter"):
        """
        Cluster data multiple times for each number of clusters (K) in the selected range and compute the average stability for each them.

        Parameters
        ----------
        adata
            Annotated data object.
        use_rep
            Key in :attr:`anndata.AnnData.obsm` to use as data to fit the clustering model. If ``None``, uses :attr:`anndata.AnnData.X`.
        """
        if use_rep not in adata.obsm:
            raise ValueError(f"{use_rep} not found in adata.obsm. If you want to use adata.X, set use_rep=None")

        X = adata.obsm[use_rep] if use_rep is not None else adata.X

        self.labels = defaultdict(list)
        self.best_models = {}

        random_state = self.model_params.pop("random_state", 0)

        if ("trainer_params" not in self.model_params) or (
            "enable_progress_bar" not in self.model_params["trainer_params"]
        ):
            self.model_params["trainer_params"] = {"enable_progress_bar": False}

        previous_stability = None
        for i in range(self.max_runs):
            print(f"Iteration {i + 1}/{self.max_runs}")
            new_labels = {}

            for k in tqdm(self.n_clusters, disable=(len(self.n_clusters) == 1)):
                logging_level = logging.getLogger("lightning.pytorch").getEffectiveLevel()
                logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
                clustering = self.model_class(n_clusters=k, random_state=i + random_state, **self.model_params)
                clustering.fit(X)
                new_labels[k] = clustering.predict(X)

                logging.getLogger("lightning.pytorch").setLevel(logging_level)

                if (k not in self.best_models.keys()) or (clustering.nll_ < self.best_models[k].nll_):
                    self.best_models[k] = clustering

            if i > 0:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    pairs = [
                        (new_labels[k], self.labels[k + 1][i])
                        for i in range(len(list(self.labels.values())[0]))
                        for k in list(self.labels.keys())[:-1]
                    ]
                    self.stability.extend(list(executor.map(lambda x: self.similarity_function(*x), pairs)))

                if previous_stability is not None:
                    stability_change = mean_absolute_percentage_error(
                        np.mean(self._mirror_stability(previous_stability), axis=1),
                        np.mean(self._mirror_stability(self.stability), axis=1),
                    )
                    if stability_change < self.convergence_tol:
                        for k, new_l in new_labels.items():
                            self.labels[k].append(new_l)
                        print(
                            f"Convergence with a change in stability of {stability_change} reached after {i + 1} iterations"
                        )
                        break

                previous_stability = deepcopy(self.stability)

            for k, new_l in new_labels.items():
                self.labels[k].append(new_l)

        if self.max_runs > 1:
            self.stability = self._mirror_stability(self.stability)
        else:
            self.stability = None

    def _mirror_stability(self, stability):
        stability = [
            stability[i : i + len(self.n_clusters) - 1] for i in range(0, len(stability), len(self.n_clusters) - 1)
        ]
        stability = list(map(list, zip(*stability)))
        return np.array([stability[i] + stability[i - 1] for i in range(1, len(stability))])

    @property
    def best_k(self) -> int:
        """The number of clusters with the highest stability."""
        if self.max_runs <= 1:
            raise ValueError("Cannot compute stability with max_runs <= 1")
        stability_mean = np.array([np.mean(self.stability[k]) for k in range(len(self.n_clusters[1:-1]))])
        best_idx = np.argmax(stability_mean)
        return self.n_clusters[best_idx + 1]

    @property
    def peaks(self) -> List[int]:
        """Find the peaks in the stability curve."""
        if self.max_runs <= 1:
            raise ValueError("Cannot compute stability with max_runs <= 1")
        stability_mean = np.array([np.mean(self.stability[k]) for k in range(len(self.n_clusters[1:-1]))])
        peaks, _ = find_peaks(stability_mean)
        return np.array(self.n_clusters[1:-1])[peaks]

    def predict(self, adata: ad.AnnData, use_rep: str = None, k: int = None) -> pd.Categorical:
        """
        Predict the labels for the data in ``use_rep`` using the fitted model.

        Parameters
        ----------
        adata
            Annotated data object.
        use_rep
            Key in :attr:`anndata.AnnData.obsm` used as data to fit the clustering model. If ``None``, uses :attr:`anndata.AnnData.obsm['X_cellcharter']` if present, otherwise :attr:`anndata.AnnData.X`.
        k
            Number of clusters to predict using the fitted model. If ``None``, the number of clusters with the highest stability will be selected. If ``max_runs > 1``, the model with the largest marginal likelihood will be used among the ones fitted on ``k``.
        """
        k = self.best_k if k is None else k
        assert k is None or k in self.n_clusters

        X = (
            adata.obsm[use_rep]
            if use_rep is not None
            else adata.obsm["X_cellcharter"] if "X_cellcharter" in adata.obsm else adata.X
        )
        return pd.Categorical(self.best_models[k].predict(X).astype(str), categories=np.arange(k).astype(str))

    @property
    def persistent_attributes(self) -> List[str]:
        """Returns the list of fitted attributes that ought to be saved and loaded. By default, this encompasses all annotations."""
        return list(self.__annotations__.keys())

    def save(self, path: PathType, best_k=False) -> None:
        """
        Saves the ClusterAutoK object and the clustering models to the provided directory using pickle.

        Parameters
        ----------
        path
            The directory to which all files should be saved.
        best_k
            Save only the best model out all number of clusters `K`. If ``false``, save the best model for each value of `K`.
        Note
        ----------
            If the dictionary returned by :func:`get_params` is not JSON-serializable, this method
            uses :mod:`pickle` which is not necessarily backwards-compatible.
        """
        path = Path(path)
        assert not path.exists() or path.is_dir(), "Estimators can only be saved to a directory."

        path.mkdir(parents=True, exist_ok=True)
        self._save_parameters(path)
        self._save_attributes(path, best_k=best_k)

    def _save_parameters(self, path: Path) -> None:
        """
        Saves the parameters of ClusterAutoK. By default, it uses JSON and falls back to :mod:`pickle`.

        Parameters
        ----------
        path
            The directory to which the parameters should be saved.
        """
        params = self.get_params()
        try:
            data = json.dumps(params, indent=4)
            with (path / "params.json").open("w+") as f:
                f.write(data)
        except TypeError:
            # warnings.warn(
            #     f"Failed to serialize parameters of `{self.__class__.__name__}` to JSON. " "Falling back to `pickle`.",
            #     stacklevel=2,
            # )
            with (path / "params.pickle").open("wb+") as f:
                pickle.dump(params, f)

    def _save_attributes(self, path: Path, best_k: bool) -> None:
        """
        Saves the attributes of ClusterAutoK. By default, it uses JSON and falls back to :mod:`pickle`.

        Parameters
        ----------
        path
            The directory to which the fitted attributed should be saved.
        best_k
            Save only the best model out all number of clusters `K`. If ``false``, save the best model for each value of `K`.
        """
        if len(self.persistent_attributes) == 0:
            return

        if best_k:
            model = self.best_models[self.best_k]
            model.save(path / "best_models" / f"{model.__class__.__name__}_k{self.best_k}")
        else:
            for k, model in self.best_models.items():
                model.save(path / "best_models" / f"{model.__class__.__name__}_k{k}")

        attributes = {
            attribute: getattr(self, attribute)
            for attribute in self.persistent_attributes
            if attribute != "best_models"
        }
        try:
            data = json.dumps(attributes, indent=4)
            with (path / "attributes.json").open("w+") as f:
                f.write(data)
        except TypeError:
            # warnings.warn(
            #     f"Failed to serialize fitted attributes of `{self.__class__.__name__}` to JSON. "
            #     "Falling back to `pickle`.",
            #     stacklevel=2,
            # )
            with (path / "attributes.pickle").open("wb+") as f:
                pickle.dump(attributes, f)

    @classmethod
    def load(cls, path: Path):
        """
        Loads the estimator and (if available) the fitted model.

        This method should only be expected to work to load an estimator that has previously been saved via :func:`save`.

        Parameters
        ----------
        path
            The directory from which to load the estimator.
        Returns
        ----------
            The loaded estimator, either fitted or not.
        """
        path = Path(path)
        assert path.is_dir(), "Estimators can only be loaded from a directory."

        model = cls._load_parameters(path)
        try:
            model._load_attributes(path)
        except FileNotFoundError:
            warnings.warn(f"Failed to read fitted attributes of `{cls.__name__}` at path '{path}'", stacklevel=2)

        return model

    @classmethod
    def _load_parameters(cls, path: Path):
        """
        Initializes this estimator by loading its parameters.

        If subclasses overwrite :func:`save_parameters`, this method should also be overwritten.
        Typically, this method should not be called directly. It is called as part of :func:`load`.

        Parameters
        ----------
        path
            The directory from which the parameters should be loaded.
        """
        json_path = path / "params.json"
        pickle_path = path / "params.pickle"

        if json_path.exists():
            with json_path.open() as f:
                params = json.load(f)
        else:
            with pickle_path.open("rb") as f:
                params = pickle.load(f)

        return cls(**params)

    def _load_attributes(self, path: Path) -> None:
        """
        Loads the fitted attributes that are stored at the fitted path.

        If subclasses overwrite :func:`save_attributes`, this method should also be overwritten.
        Typically, this method should not be called directly. It is called as part of :func:`load`.

        Parameters
        ----------
        path
            The directory from which the parameters should be loaded.
        Raises
        ----------
        FileNotFoundError
            If the no fitted attributes have been stored.
        """
        json_path = path / "attributes.json"
        pickle_path = path / "attributes.pickle"

        if json_path.exists():
            with json_path.open() as f:
                self.set_params(json.load(f))
        else:
            with pickle_path.open("rb") as f:
                self.set_params(pickle.load(f))

        self.best_models = {}
        for model_dir in os.listdir(path / "best_models"):
            model = self.model_class()
            model = model.load(path / "best_models" / model_dir)

            self.best_models[model.n_clusters] = model

    def get_params(self) -> Dict[str, Any]:
        """
        Returns the estimator's parameters as passed to the initializer.

        Args
        ----------
        deep
            Ignored. For Scikit-learn compatibility.
        Returns
        ----------
            The mapping from init parameters to values.
        """
        signature = inspect.signature(self.__class__.__init__)
        parameters = [p.name for p in signature.parameters.values() if p.name != "self"]
        return {p: getattr(self, p) for p in parameters}

    def set_params(self, values: Dict[str, Any]):
        """
        Sets the provided values. The estimator is returned as well, but the estimator on which this function is called is also modified.

        Parameters
        ----------
        values
            The values to set.
        Returns
        ----------
            The estimator where the values have been set.
        """
        for key, value in values.items():
            setattr(self, key, value)
        return self
