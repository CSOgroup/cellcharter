from __future__ import annotations

import os
from typing import Optional

from anndata import AnnData, read
from torch import nn

try:
    from scarches.models import TRVAE as scaTRVAE
    from scarches.models import trVAE
    from scarches.models.base._utils import _validate_var_names
except ImportError:

    class TRVAE:  # noqa: D101
        def __init__(
            self,
            adata: AnnData,
            condition_key: str = None,
            conditions: Optional[list] = None,
            hidden_layer_sizes: list | tuple = (256, 64),
            latent_dim: int = 10,
            dr_rate: float = 0.05,
            use_mmd: bool = True,
            mmd_on: str = "z",
            mmd_boundary: Optional[int] = None,
            recon_loss: Optional[str] = "nb",
            beta: float = 1,
            use_bn: bool = False,
            use_ln: bool = True,
        ):
            raise ImportError("scarches is not installed. Please install scarches to use this method.")

        @classmethod
        def load(cls, dir_path: str, adata: Optional[AnnData] = None, map_location: Optional[str] = None):  # noqa: D102
            raise ImportError("scarches is not installed. Please install scarches to use this method.")

else:

    class TRVAE(scaTRVAE):
        r"""
        scArches\'s trVAE model adapted to image-based proteomics data.

        The last ReLU layer of the neural network is removed to allow for continuous and real output values

        Parameters
        ----------
        adata: `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
        condition_key: String
            column name of conditions in `adata.obs` data frame.
        conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
        hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
        latent_dim: Integer
            Bottleneck layer (z) size.
        dr_rate: Float
            Dropout rate applied to all layers, if `dr_rate==0` no dropout will be applied.
        use_mmd: Boolean
            If 'True' an additional MMD loss will be calculated on the latent dim. 'z' or the first decoder layer 'y'.
        mmd_on: String
            Choose on which layer MMD loss will be calculated on if 'use_mmd=True': 'z' for latent dim or 'y' for first
            decoder layer.
        mmd_boundary: Integer or None
            Choose on how many conditions the MMD loss should be calculated on. If 'None' MMD will be calculated on all
            conditions.
        recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse', 'nb' or 'zinb'.
        beta: Float
            Scaling Factor for MMD loss
        use_bn: Boolean
            If `True` batch normalization will be applied to layers.
        use_ln: Boolean
            If `True` layer normalization will be applied to layers.
        """

        def __init__(
            self,
            adata: AnnData,
            condition_key: str = None,
            conditions: Optional[list] = None,
            hidden_layer_sizes: list | tuple = (256, 64),
            latent_dim: int = 10,
            dr_rate: float = 0.05,
            use_mmd: bool = True,
            mmd_on: str = "z",
            mmd_boundary: Optional[int] = None,
            recon_loss: Optional[str] = "mse",
            beta: float = 1,
            use_bn: bool = False,
            use_ln: bool = True,
        ):

            self.adata = adata

            self.condition_key_ = condition_key

            if conditions is None:
                if condition_key is not None:
                    self.conditions_ = adata.obs[condition_key].unique().tolist()
                else:
                    self.conditions_ = []
            else:
                self.conditions_ = conditions

            self.hidden_layer_sizes_ = hidden_layer_sizes
            self.latent_dim_ = latent_dim
            self.dr_rate_ = dr_rate
            self.use_mmd_ = use_mmd
            self.mmd_on_ = mmd_on
            self.mmd_boundary_ = mmd_boundary
            self.recon_loss_ = recon_loss
            self.beta_ = beta
            self.use_bn_ = use_bn
            self.use_ln_ = use_ln

            self.input_dim_ = adata.n_vars

            self.model = trVAE(
                self.input_dim_,
                self.conditions_,
                self.hidden_layer_sizes_,
                self.latent_dim_,
                self.dr_rate_,
                self.use_mmd_,
                self.mmd_on_,
                self.mmd_boundary_,
                self.recon_loss_,
                self.beta_,
                self.use_bn_,
                self.use_ln_,
            )

            decoder_layer_sizes = self.model.hidden_layer_sizes.copy()
            decoder_layer_sizes.reverse()
            decoder_layer_sizes.append(self.model.input_dim)

            self.model.decoder.recon_decoder = nn.Linear(decoder_layer_sizes[-2], decoder_layer_sizes[-1])

            self.is_trained_ = False

            self.trainer = None

        @classmethod
        def load(cls, dir_path: str, adata: Optional[AnnData] = None, map_location: Optional[str] = None):
            """
            Instantiate a model from the saved output.

            Parameters
            ----------
            dir_path
                Path to saved outputs.
            adata
                AnnData object.
                If None, will check for and load anndata saved with the model.
            map_location
                Location where all tensors should be loaded (e.g., `torch.device('cpu')`)
            Returns
            -------
                Model with loaded state dictionaries.
            """
            adata_path = os.path.join(dir_path, "adata.h5ad")

            load_adata = adata is None

            if os.path.exists(adata_path) and load_adata:
                adata = read(adata_path)
            elif not os.path.exists(adata_path) and load_adata:
                raise ValueError("Save path contains no saved anndata and no adata was passed.")

            attr_dict, model_state_dict, var_names = cls._load_params(dir_path, map_location=map_location)

            # Overwrite adata with new genes
            adata = _validate_var_names(adata, var_names)

            cls._validate_adata(adata, attr_dict)
            init_params = cls._get_init_params_from_dict(attr_dict)

            model = cls(adata, **init_params)
            model.model.load_state_dict(model_state_dict)
            model.model.eval()

            model.is_trained_ = attr_dict["is_trained_"]

            return model
