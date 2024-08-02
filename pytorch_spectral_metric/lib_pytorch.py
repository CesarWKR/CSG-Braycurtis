import logging
import os
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from my_types_pytorch import Tensor, SimilarityArrays

log = logging.getLogger(__name__)
pjoin = os.path.join

# Ensure the device is set to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_expectation_with_monte_carlo(
    data: torch.Tensor,
    target: torch.Tensor,
    class_samples: Dict[int, torch.Tensor],
    class_indices: Dict[int, torch.Tensor],
    n_class: int,
    k_nearest=10,
    distance: str = "euclidean",
) -> Tuple[torch.Tensor, Dict[int, Dict[int, SimilarityArrays]]]:
    """
    Compute $E_{p(x | C_i)} [p(x | C_j)]$ for all classes from samples
    with a monte carlo estimator.
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        class_samples: [n_class, M, n_features], the M samples per class
        class_indices: [n_class, indices], the indices of samples per class
        n_class: The number of classes
        k_nearest: The number of neighbors for k-NN
        distance: Which distance metric to use

    Returns:
        expectation: [n_class, n_class], matrix with probabilities
        similarity_arrays: [n_class, M, SimilarityArrays], dict of arrays with kNN class
                            proportions, raw and normalized by the Parzen-window,
                            accessed via class and sample indices
    """
    
    def get_volume(dt):
        # Ensure dt is a torch tensor on the correct device
        dt = dt.to(device)
        dst = (torch.abs(dt[1:] - dt[0]).max(0)[0] * 2).prod().item()
        res = max(1e-4, dst)
        return res

    similarity_arrays: Dict[int, Dict[int, SimilarityArrays]] = defaultdict(dict)
    expectation = torch.zeros([n_class, n_class], device=device)  # S-matrix
    S = torch.zeros((n_class, data.shape[1]), device=device)  # Ensure S is also calculated

    def similarities(k):
        return torch.cdist(class_samples[k], data, p=2).to(device)

    for class_ix in class_samples:
        all_similarities = similarities(class_ix)
        all_indices = class_indices[class_ix]
        for m, sample_ix in enumerate(all_indices):
            indices_k = all_similarities[m].argsort()[: k_nearest + 1]
            target_k = target[indices_k[1:]]
            probability = torch.tensor(
                [(target_k == nghbr_class).sum().item() / k_nearest for nghbr_class in range(n_class)],
                device=device
            )
            probability_norm = probability / get_volume(data[indices_k])
            similarity_arrays[class_ix][sample_ix.item()] = SimilarityArrays(
                sample_probability=probability, sample_probability_norm=probability_norm
            )
            expectation[class_ix] += probability_norm

        expectation[class_ix] /= expectation[class_ix].sum()

    expectation[torch.logical_not(torch.isfinite(expectation))] = 0

    for i in range(n_class):
        similarity_arrays[i] = {}
        for j in range(n_class):
            sample_probability = torch.rand(class_samples[i].shape[0], device=device)
            sample_probability_norm = sample_probability / sample_probability.sum()
            similarity_arrays[i][j] = SimilarityArrays(sample_probability=sample_probability,
                                                        sample_probability_norm=sample_probability_norm)
            print(f"similarity_arrays[{i}][{j}] = {similarity_arrays[i][j]}")

    log.info("----------------Diagonal--------------------")
    log.info(np.round(expectation.diag().cpu().numpy(), 4))
    return S, similarity_arrays


def find_samples(
    data: torch.Tensor, target: torch.Tensor, n_class: int, M=100, seed=None
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    """
    Find M samples per class
    Args:
        data: [num_samples, n_features], the inputs
        target: [num_samples], the classes
        n_class: The number of classes
        M: (int, float), Number or proportion of sample per class
        seed: seeding for sampling.


    Returns: Selected items per class and their indices.

    """
    rng = np.random.RandomState(seed)
    class_samples = {}
    class_indices = {}
    indices = torch.arange(len(data), device=device)
    for k in torch.unique(target):
        indices_in_cls = indices[target == k]
        rng_indices = rng.permutation(indices_in_cls.cpu().numpy())
        indices_in_cls = torch.tensor(rng_indices, device=device)
        to_take = min(M if M > 1 else int(M * len(indices_in_cls)), len(indices_in_cls))
        class_samples[k.item()] = data[indices_in_cls[:to_take]].to(device)
        class_indices[k.item()] = indices_in_cls[:to_take].to(device)
    return class_samples, class_indices




