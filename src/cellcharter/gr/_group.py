from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from squidpy._docs import d
from tqdm import tqdm


def _proportion(obs, id_key, val_key, normalize=True):
    df = pd.pivot(obs[[id_key, val_key]].value_counts().reset_index(), index=id_key, columns=val_key)
    df[df.isna()] = 0
    df.columns = df.columns.droplevel(0)
    if normalize:
        return df.div(df.sum(axis=1), axis=0)
    else:
        return df


def _observed_permuted(annotations, group_key, label_key):
    annotations[group_key] = annotations[group_key].sample(frac=1).reset_index(drop=True).values
    return _proportion(annotations, id_key=label_key, val_key=group_key).reindex().T


def _enrichment(observed, expected, log=True):
    enrichment = observed.div(expected, axis="index", level=0)

    if log:
        enrichment = np.log2(enrichment)
    enrichment = enrichment.fillna(enrichment.min())
    return enrichment


def _empirical_pvalues(observed, expected):
    pvalues = np.zeros(observed.shape)
    pvalues[observed.values > 0] = (
        1 - np.sum(expected[:, observed.values > 0] < observed.values[observed.values > 0], axis=0) / expected.shape[0]
    )
    pvalues[observed.values < 0] = (
        1 - np.sum(expected[:, observed.values < 0] > observed.values[observed.values < 0], axis=0) / expected.shape[0]
    )
    return pd.DataFrame(pvalues, columns=observed.columns, index=observed.index)


@d.dedent
def enrichment(
    adata: AnnData,
    group_key: str,
    label_key: str,
    pvalues: bool = False,
    n_perms: int = 1000,
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
    pvalues
        If `True`, compute empirical p-values by permutation. It will result in a slower computation.
    n_perms
        Number of permutations to compute empirical p-values.
    log
        If `True` use log2 fold change, otherwise use fold change.
    observed_expected
        If `True`, return also the observed and expected proportions.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:
        - ``'enrichment'`` - the enrichment values.
        - ``'pvalue'`` - the enrichment pvalues (if `pvalues is True`).
        - ``'observed'`` - the observed proportions (if `observed_expected is True`).
        - ``'expected'`` - the expected proportions (if `observed_expected is True`).

    Otherwise, modifies the ``adata`` with the following keys:
        - :attr:`anndata.AnnData.uns` ``['{group_key}_{label_key}_nhood_enrichment']`` - the above mentioned dict.
        - :attr:`anndata.AnnData.uns` ``['{group_key}_{label_key}_nhood_enrichment']['params']`` - the parameters used.
    """
    observed = _proportion(adata.obs, id_key=label_key, val_key=group_key).reindex().T
    observed[observed.isna()] = 0
    if not pvalues:
        expected = adata.obs[group_key].value_counts() / adata.shape[0]
        # Repeat over the number of labels
        expected = pd.concat([expected] * len(observed.columns), axis=1, keys=observed.columns)
    else:
        annotations = adata.obs.copy()

        expected = [_observed_permuted(annotations, group_key, label_key) for _ in tqdm(range(n_perms))]
        expected = np.stack(expected, axis=0)

        print(expected.shape)

        empirical_pvalues = _empirical_pvalues(observed, expected)

        expected = np.mean(expected, axis=0)
        expected = pd.DataFrame(expected, columns=observed.columns, index=observed.index)

    print(expected)
    enrichment = _enrichment(observed, expected, log=log)

    result = {"enrichment": enrichment}

    if observed_expected:
        result["observed"] = observed
        result["expected"] = expected

    if pvalues:
        result["pvalue"] = empirical_pvalues

    if copy:
        return result
    else:
        adata.uns[f"{group_key}_{label_key}_enrichment"] = result
        adata.uns[f"{group_key}_{label_key}_enrichment"]["params"] = {"log": log}
