from __future__ import annotations

from typing import List, Tuple, cast

import torch
from lightkit.data import DataLoader, TensorLike, collate_tensor, dataset_from_tensors
from pycave.bayes import GaussianMixture as PyCaveGaussianMixture
from pycave.bayes.gmm.lightning_module import GaussianMixtureLightningModule
from pycave.bayes.gmm.model import GaussianMixtureModel
from pytorch_lightning import Trainer
from scanpy._utils import AnyRandom
from sklearn.cluster import KMeans, MiniBatchKMeans


class GaussianMixture(PyCaveGaussianMixture):
    """
    Adapted version of GaussianMixture clustering model from the `PyCave <https://pycave.borchero.com/index.html>`_ library.

    Parameters
    ----------
    n_clusters
        The number of components in the GMM. The dimensionality of each component is automatically inferred from the data.
    covariance_type
        The type of covariance to assume for all Gaussian components.
    init_strategy
        The strategy for initializing component means and covariances.
    init_means
        An optional initial guess for the means of the components. If provided,
        must be a tensor of shape ``[num_components, num_features]``. If this is given,
        the ``init_strategy`` is ignored and the means are handled as if K-means
        initialization has been run.
    convergence_tolerance
        The change in the per-datapoint negative log-likelihood which
        implies that training has converged.
    covariance_regularization
        A small value which is added to the diagonal of the
        covariance matrix to ensure that it is positive semi-definite.
    batch_size: The batch size to use when fitting the model. If not provided, the full
        data will be used as a single batch. Set this if the full data does not fit into
        memory.
    trainer_params
        Initialization parameters to use when initializing a PyTorch Lightning
        trainer. By default, it disables various stdout logs unless PyCave is configured to
        do verbose logging. Checkpointing and logging are disabled regardless of the log
        level. This estimator further sets the following overridable defaults:
        - ``max_epochs=100``.
    random_state
        Initialization seed.

    """

    #: The fitted PyTorch module with all estimated parameters.
    model_: GaussianMixtureModel
    #: A boolean indicating whether the model converged during training.
    converged_: bool
    #: The number of iterations the model was fitted for, excluding initialization.
    num_iter_: int
    #: The average per-datapoint negative log-likelihood at the last training step.
    nll_: float

    def __init__(
        self,
        n_clusters: int = 1,
        *,
        covariance_type: str = "full",
        init_strategy: str = "sklearn",
        init_means: torch.Tensor = None,
        convergence_tolerance: float = 0.001,
        covariance_regularization: float = 1e-06,
        batch_size: int = None,
        trainer_params: dict = None,
        random_state: AnyRandom = 0,
    ):
        super().__init__(
            num_components=n_clusters,
            covariance_type=covariance_type,
            init_strategy=init_strategy,
            init_means=init_means,
            convergence_tolerance=convergence_tolerance,
            covariance_regularization=covariance_regularization,
            batch_size=batch_size,
            trainer_params=trainer_params,
        )
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, data: TensorLike) -> GaussianMixture:
        """
        Fits the Gaussian mixture on the provided data, estimating component priors, means and covariances. Parameters are estimated using the EM algorithm.

        Parameters
        ----------
        data
            The tabular data to fit on. The dimensionality of the Gaussian mixture is automatically inferred from this data.
        Returns
        ----------
            The fitted Gaussian mixture.
        """
        if self.init_strategy == "sklearn":
            if self.batch_size is None:
                kmeans = KMeans(n_clusters=self.num_components, random_state=self.random_state, n_init=1)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.num_components, random_state=self.random_state, n_init=1, batch_size=self.batch_size
                )
            kmeans.fit(data)
            self.init_means = torch.tensor(kmeans.cluster_centers_).float()
        return self._fit(data)

    def _fit(self, data) -> GaussianMixture:
        try:
            return super().fit(data)
        except torch._C._LinAlgError:
            self.covariance_regularization *= 10
            return self._fit(data)

    def predict(self, data: TensorLike) -> torch.Tensor:
        """
        Computes the most likely components for each of the provided datapoints.

        Parameters
        ----------
        data
            The datapoints for which to obtain the most likely components.
        Returns
        ----------
            A tensor of shape ``[num_datapoints]`` with the indices of the most likely components.
        Note
        ----------
            Use :func:`predict_proba` to obtain probabilities for each component instead of the
            most likely component only.
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        return super().predict(data).numpy()

    def predict_proba(self, data: TensorLike) -> torch.Tensor:
        """
        Computes a distribution over the components for each of the provided datapoints.

        Parameters
        ----------
        data
            The datapoints for which to compute the component assignment probabilities.
        Returns
        ----------
            A tensor of shape ``[num_datapoints, num_components]`` with the assignment
            probabilities for each component and datapoint. Note that each row of the vector sums
            to 1, i.e. the returned tensor provides a proper distribution over the components for
            each datapoint.
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        result = Trainer(logger=False).predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.cat([x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])

    def score_samples(self, data: TensorLike) -> torch.Tensor:
        """
        Computes the negative log-likelihood (NLL) of each of the provided datapoints.

        Parameters
        ----------
        data
            The datapoints for which to compute the NLL.
        Returns
        ----------
            A tensor of shape ``[num_datapoints]`` with the NLL for each datapoint.
        Attention
        ----------
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        result = Trainer(logger=False).predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.stack([x[1] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])
