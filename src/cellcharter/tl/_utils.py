from copy import deepcopy

from sklearn.metrics import fowlkes_mallows_score


def _stability_onesided(labels, robustness, new_labels):
    # Similarities between the existing labels at K and the new labels at K+1
    for _, (k, new_l) in enumerate(new_labels.items()):
        if k - 1 in labels:
            for old_l in labels[k - 1]:
                robustness[k - 1].append(fowlkes_mallows_score(new_l, old_l))

    for k, new_l in new_labels.items():
        labels[k].append(new_l)

    # Similarities between the new labels at K and all the labels at K+1
    for _, (k, new_l) in enumerate(new_labels.items()):
        if k + 1 in labels:
            for old_l in labels[k + 1]:
                robustness[k].append(fowlkes_mallows_score(new_l, old_l))
    return labels, robustness


# robustness[k] is the robustness between k and k+1
# We need to append robustness between k-1 and k
def _stability_twosided(robustness):
    robustness_twosided = deepcopy(robustness)

    for k in robustness.keys():
        robustness_twosided[k + 1].extend(robustness[k])

    return robustness_twosided
