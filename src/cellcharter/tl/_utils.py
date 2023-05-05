from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score


def _stability(labels, max_runs, similarity_function=adjusted_rand_score, n_jobs=-1):
    num_labels = len(labels)
    num_combinations = max_runs * (max_runs - 1) // 2

    stabilities = Parallel(n_jobs=n_jobs)(
        delayed(similarity_function)(labels[k][i], labels[k + 1][j])
        for k in range(1, num_labels)
        for i, j in combinations(range(max_runs), 2)
    )

    # Transform test into a list of chunks of size num_combinations
    stabilities = [stabilities[i : i + num_combinations] for i in range(0, len(stabilities), num_combinations)]

    # Append to every element of test the previous element
    stabilities = [stabilities[i] + stabilities[i - 1] for i in range(1, len(stabilities))]

    return np.array(stabilities)
