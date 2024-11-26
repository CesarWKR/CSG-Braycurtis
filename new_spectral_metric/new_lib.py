import logging
import os
from collections import defaultdict
from typing import Tuple, Dict

import numpy as np
import scipy as sp
import scipy.stats
from sklearn.metrics import pairwise_distances
from new_spectral_metric.my_types import Array, SimilarityArrays
from .my_types import Array, SimilarityArrays

log = logging.getLogger(__name__)
pjoin = os.path.join

def compute_expectation_with_monte_carlo(
    data: Array,
    target: Array,
    class_samples: Dict[int, Array],
    class_indices: Dict[int, Array],
    n_class: int,
    k_nearest=10,
    distance: str = "euclidean",
) -> Tuple[Array, Dict[int, Dict[int, SimilarityArrays]]]:
    def get_volume(dt):
        dst = (np.abs(dt[1:] - dt[0]).max(0) * 2).prod()
        res = max(1e-4, dst)
        return res

    similarity_arrays: Dict[int, Dict[int, SimilarityArrays]] = defaultdict(dict)
    expectation = np.zeros([n_class, n_class])  # S-matrix

    similarities = lambda k: np.array(pairwise_distances(class_samples[k], data, metric=distance))

    for class_ix in class_samples:
        all_similarities = similarities(class_ix)  # Distance arrays for all class samples
        all_indices = class_indices[class_ix]  # Indices for all class samples
        for m, sample_ix in enumerate(all_indices):
            indices_k = all_similarities[m].argsort()[: k_nearest + 1]  # kNN indices (incl self)
            target_k = np.array(target)[indices_k[1:]]  # kNN class labels (self is dropped)
            probability = np.array(
                [(target_k == nghbr_class).sum() / k_nearest for nghbr_class in range(n_class)]
            )
            probability_norm = probability / get_volume(data[indices_k])  # Parzen-window normalized
            similarity_arrays[class_ix][sample_ix] = SimilarityArrays(
                sample_probability=probability, sample_probability_norm=probability_norm
            )
            expectation[class_ix] += probability_norm

        expectation[class_ix] /= expectation[class_ix].sum()

    expectation[np.logical_not(np.isfinite(expectation))] = 0

    log.info("----------------Diagonal--------------------")
    log.info(np.round(np.diagonal(expectation), 4))
    return expectation, similarity_arrays

def find_samples(
    data: np.ndarray, target: np.ndarray, n_class: int, M=100, seed=None
) -> Tuple[Dict[int, Array], Dict[int, Array]]:
    rng = np.random.RandomState(seed)
    class_samples = {}
    class_indices = {}
    indices = np.arange(len(data))
    for k in np.unique(target):
        indices_in_cls = rng.permutation(indices[target == k])
        to_take = min(M if M > 1 else int(M * len(indices_in_cls)), len(indices_in_cls))
        class_samples[k] = data[indices_in_cls[:to_take]]
        class_indices[k] = indices_in_cls[:to_take]
    return class_samples, class_indices

def get_cummax(eigens: np.ndarray) -> Tuple[float, float]:
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)
        return m, m - h, m + h

    grads = eigens[:, 1:] - eigens[:, :-1]
    ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
    cumsums = np.maximum.accumulate(ratios, -1).sum(1)
    mu, lb, ub = mean_confidence_interval(cumsums)
    return mu, ub - mu
