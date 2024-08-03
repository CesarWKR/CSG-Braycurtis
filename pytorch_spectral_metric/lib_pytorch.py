import logging
import os
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from my_types_pytorch import Array, SimilarityArrays

log = logging.getLogger(__name__)
pjoin = os.path.join

# Ensure the device is set to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        dt = dt.cpu().numpy()  # Convert to NumPy array for this specific operation  
        dst = (abs(dt[1:] - dt[0]).max(0) * 2).prod()  
        res = max(1e-4, dst)  
        return torch.tensor(res)  # Convert back to Tensor  

    similarity_arrays: Dict[int, Dict[int, SimilarityArrays]] = defaultdict(dict)  
    expectation = torch.zeros([n_class, n_class])  # S-matrix  

    similarities = lambda k: torch.tensor(pairwise_distances(class_samples[k].cpu().numpy(), data.cpu().numpy(), metric=distance))  

    for class_ix in class_samples:  
        all_similarities = similarities(class_ix)  # Distance arrays for all class samples  
        all_indices = class_indices[class_ix]  # Indices for all class samples  
        for m, sample_ix in enumerate(all_indices):  
            indices_k = all_similarities[m].argsort()[: k_nearest + 1]  # kNN indices (incl self)  
            target_k = torch.tensor(target.cpu().numpy())[indices_k[1:]]  # kNN class labels (self is dropped)  
            probability = torch.tensor(  
                [(target_k == nghbr_class).sum().item() / k_nearest for nghbr_class in range(n_class)]  
            )  
            probability_norm = probability / get_volume(data[indices_k])  # Parzen-window normalized  
            similarity_arrays[class_ix][sample_ix] = SimilarityArrays(  
                sample_probability=probability, sample_probability_norm=probability_norm  
            )  
            expectation[class_ix] += probability_norm  

        expectation[class_ix] /= expectation[class_ix].sum()  

    expectation[torch.logical_not(torch.isfinite(expectation))] = 0  

    log.info("----------------Diagonal--------------------")  
    log.info(torch.round(torch.diagonal(expectation), decimals=4))  
    return expectation, similarity_arrays  

def find_samples(  
    data: Array, target: Array, n_class: int, M=100, seed=None  
) -> Tuple[Dict[int, Array], Dict[int, Array]]:  
    rng = torch.Generator()  
    if seed is not None:  
        rng.manual_seed(seed)  
    
    class_samples = {}  
    class_indices = {}  
    indices = torch.arange(len(data), generator=rng)  
    for k in torch.unique(target):  
        indices_in_cls = indices[torch.nonzero(target == k).squeeze()]  
        if M > 1:  
            to_take = min(M, len(indices_in_cls))  
        else:  
            to_take = min(int(M * len(indices_in_cls)), len(indices_in_cls))  
        indices_in_cls = indices_in_cls[torch.randperm(len(indices_in_cls), generator=rng)[:to_take]]  
        class_samples[k.item()] = data[indices_in_cls]  
        class_indices[k.item()] = indices_in_cls  
    return class_samples, class_indices  

def get_cummax(eigens: Array) -> Tuple[float, float]:  
    def mean_confidence_interval(data, confidence=0.95):  
        data = data.numpy()  # Convert to NumPy array for SciPy operations  
        a = 1.0 * data  
        n = len(a)  
        m, se = torch.tensor(np.mean(a)), torch.tensor(sp.stats.sem(a))  
        h = se * torch.tensor(sp.stats.t.ppf((1 + confidence) / 2.0, n - 1))  
        return m, m - h, m + h  

    grads = eigens[:, 1:] - eigens[:, :-1]  
    ratios = grads / (torch.arange(1, grads.shape[-1] + 1).flip(0) + 1)  
    cumsums = torch.cumsum(torch.max(ratios, dim=-1).values, dim=0).sum(1)  
    mu, lb, ub = mean_confidence_interval(cumsums)  
    return mu.item(), (ub - mu).item()  



