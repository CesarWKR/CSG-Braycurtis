import logging  
import os  
from collections import defaultdict  
from typing import Tuple, Dict  

import cupy as cp  
import numpy as np  
import scipy as sp  
import scipy.stats  
from sklearn.metrics import pairwise_distances  
import torch  
from pytorch_spectral_metric.my_types_pytorch import SimilarityArrays  

log = logging.getLogger(__name__)  
pjoin = os.path.join  

def compute_expectation_with_monte_carlo(  
    data: cp.ndarray,  
    target: cp.ndarray,  
    class_samples: Dict[int, cp.ndarray],  
    class_indices: Dict[int, cp.ndarray],  
    n_class: int,  
    k_nearest=10,  
    distance: str = "euclidean",  
) -> Tuple[cp.ndarray, Dict[int, Dict[int, 'SimilarityArrays']]]:  
    
    def get_volume(dt):  
        dst = (cp.abs(dt[1:] - dt[0]).max(0) * 2).prod()  
        return max(1e-4, dst)  

    similarity_arrays = defaultdict(dict)  
    expectation = cp.zeros([n_class, n_class])  

    def similarities(k):  
        return cp.array(pairwise_distances(cp.asnumpy(class_samples[k]), cp.asnumpy(data), metric=distance))  

    for class_ix in class_samples:  
        all_similarities = similarities(class_ix)  
        all_indices = class_indices[class_ix]  

        for m, sample_ix in enumerate(all_indices):  
            indices_k = cp.argsort(all_similarities[m])[: k_nearest + 1]  
            target_k = target[indices_k[1:]]  # CuPy target no requiere conversión explícita aquí  
            
            probability = cp.array([(target_k == nghbr_class).sum() / k_nearest for nghbr_class in range(n_class)])  
            probability_norm = probability / get_volume(data[indices_k])  

            similarity_arrays[class_ix][sample_ix] = {  
                "sample_probability": probability,  
                "sample_probability_norm": probability_norm  
            }  
            expectation[class_ix] += probability_norm  

        expectation[class_ix] /= expectation[class_ix].sum()  

    expectation[~cp.isfinite(expectation)] = 0  

    log.info("----------------Diagonal--------------------")  
    log.info(cp.asnumpy(cp.round(cp.diagonal(expectation), 4)))  
    return expectation, similarity_arrays  

def find_samples(  
    data: cp.ndarray, target: cp.ndarray, n_class: int, M=100, seed=None  
) -> Tuple[Dict[int, cp.ndarray], Dict[int, cp.ndarray]]:  
    rng = np.random.default_rng(seed)  
    class_samples = {}  
    class_indices = {}  
    
    indices = cp.arange(len(data))  
    
    for k in cp.unique(target):  
        indices_in_cls = rng.permutation(cp.asnumpy(indices[target == k].get()))  
        to_take = min(M if M > 1 else int(M * len(indices_in_cls)), len(indices_in_cls))  
        class_samples[k] = data[indices_in_cls[:to_take]]  
        class_indices[k] = indices_in_cls[:to_take]  
    
    return class_samples, class_indices  

def get_cummax(eigens: cp.ndarray) -> Tuple[float, float]:  
    def mean_confidence_interval(data, confidence=0.95):  
        a = 1.0 * cp.asnumpy(data)  
        n = len(a)  
        m, se = np.mean(a), sp.stats.sem(a)  
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)  
        return m, m - h, m + h  

    grads = eigens[:, 1:] - eigens[:, :-1]  
    ratios = grads / (cp.arange(grads.shape[-1], 0, -1) + 1)  
    cumsums = cp.maximum.accumulate(ratios, -1).sum(1)  
    mu, lb, ub = mean_confidence_interval(cumsums)  
    return mu, ub - mu


