from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from squidpy._docs import d


def _proportion(adata, id_key, val_key, normalize=True):
    df = pd.pivot(adata.obs[[id_key, val_key]].value_counts().reset_index(), index=id_key, columns=val_key)
    df[df.isna()] = 0
    df.columns = df.columns.droplevel(0)
    if normalize:
        return df.div(df.sum(axis=1), axis=0)
    else:
        return df


def _enrichment(observed, expected, log=True):
    enrichment = observed.div(expected, axis="index", level=0)

    if log:
        enrichment = np.log2(enrichment)
    enrichment = enrichment.fillna(enrichment.min())
    return enrichment


@d.dedent
def enrichment(
    adata: AnnData,
    group_key: str,
    label_key: str,
    log: bool = True,
    observed_expected: bool = False,
    copy: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Compute the enrichment of `label_key` in `group_key`.

    Parameters
    ----------
    %(adata)s
    group_key
        Key in :attr:`anndata.AnnData.obs` where groups are stored.
    label_key
        Key in :attr:`anndata.AnnData.obs` where labels are stored.
    log
        If `True` use log2 fold change, otherwise use fold change.
    observed_expected
        If `True`, return also the observed and expected proportions.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:
        - ``'enrichment'`` - the enrichment values.
        - ``'observed'`` - the observed proportions (if `observed_expected is True`).
        - ``'expected'`` - the expected proportions (if `observed_expected is True`).

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.uns` ``['{group_key}_{label_key}_nhood_enrichment']`` - the above mentioned dict.
        - :attr:`anndata.AnnData.uns` ``['{group_key}_{label_key}_nhood_enrichment']['params']`` - the parameters used.
    """
    observed = _proportion(adata, id_key=label_key, val_key=group_key).reindex().T
    observed[observed.isna()] = 0
    expected = adata.obs[group_key].value_counts() / adata.shape[0]

    enrichment = _enrichment(observed, expected, log=log)

    result = {"enrichment": enrichment}

    if observed_expected:
        result["observed"] = observed
        result["expected"] = expected

    if copy:
        return result
    else:
        adata.uns[f"{group_key}_{label_key}_enrichment"] = result
        adata.uns[f"{group_key}_{label_key}_enrichment"]["params"] = {"log": log}
