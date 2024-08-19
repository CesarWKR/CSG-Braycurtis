import logging  
import os  
from collections import defaultdict  
from typing import Tuple, Dict  

import cupy as cp  
import scipy as sp  
import scipy.stats  
from sklearn.metrics import pairwise_distances  
import torch  

# Define tus tipos aquí como en el módulo original  
# Array y SimilarityArrays deben ser definidos o importados si están en otro archivo.  
# Aquí asumimos que son análogos a np.ndarray y algunas estructuras de datos.  

log = logging.getLogger(__name__)  
pjoin = os.path.join  

def compute_expectation_with_monte_carlo(  
    data: cp.ndarray,  # Usamos Cupy para el arreglo de datos  
    target: cp.ndarray,  
    class_samples: Dict[int, cp.ndarray],  
    class_indices: Dict[int, cp.ndarray],  
    n_class: int,  
    k_nearest=10,  
    distance: str = "euclidean",  
) -> Tuple[cp.ndarray, Dict[int, Dict[int, 'SimilarityArrays']]]:  
    
    def get_volume(dt):  
        # Convertimos a numpy para operaciones específicas de numpy que CuPy aún no soporte  
        dst = (cp.abs(dt[1:] - dt[0]).max(0) * 2).prod()  
        res = max(1e-4, dst)  
        return res  

    similarity_arrays = defaultdict(dict)  
    expectation = cp.zeros([n_class, n_class])  # S-matrix en CuPy  

    # Distancias precomputadas utilizando `sklearn.metrics.pairwise_distances`  
    def similarities(k):  
        # Se puede reemplazar por una distancia compatible con GPU si se provee soporte futuro  
        return cp.array(pairwise_distances(cp.asnumpy(class_samples[k]), cp.asnumpy(data), metric=distance))  

    for class_ix in class_samples:  
        all_similarities = similarities(class_ix)  # Distancias para todas las muestras de clase  
        all_indices = class_indices[class_ix]  # Índices para todas las muestras de clase  

        for m, sample_ix in enumerate(all_indices):  
            indices_k = cp.argsort(all_similarities[m])[: k_nearest + 1]  # Índices kNN (incl. mismo)  
            
            # Convertimos temporalmente a numpy para manejo de mirrors como target  
            target_k = cp.asarray(target)[indices_k[1:]]  # Clases kNN (descartamos self)  
            
            # Calcula probabilidades  
            probability = cp.array(  
                [(target_k == nghbr_class).sum() / k_nearest for nghbr_class in range(n_class)]  
            )  
            probability_norm = probability / get_volume(data[indices_k])  # Parzen-window normalizado  
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
    # RandomState lo manteneré en numpy por simplicidad y asegurar pseudorandom  
    rng = np.random.RandomState(seed)  
    class_samples = {}  
    class_indices = {}  
    
    indices = cp.arange(len(data))  
    for k in cp.unique(target):  
        indices_in_cls = rng.permutation(cp.asnumpy(indices[target == k]))  
        to_take = min(M if M > 1 else int(M * len(indices_in_cls)), len(indices_in_cls))  
        class_samples[k] = data[indices_in_cls[:to_take]]  
        class_indices[k] = indices_in_cls[:to_take]  
    
    return class_samples, class_indices  

def get_cummax(eigens: cp.ndarray) -> Tuple[float, float]:  
    def mean_confidence_interval(data, confidence=0.95):  
        a = 1.0 * cp.asnumpy(data)  # Convertimos a numpy para confianza  
        n = len(a)  
        m, se = np.mean(a), sp.stats.sem(a)  
        h = se * sp.stats.t._ppf((1 + confidence) / 2.0, n - 1)  
        return m, m - h, m + h  

    grads = eigens[:, 1:] - eigens[:, :-1]  
    ratios = grads / (cp.arange(grads.shape[-1], 0, -1) + 1)  
    cumsums = cp.maximum.accumulate(ratios, -1).sum(1)  
    mu, lb, ub = mean_confidence_interval(cumsums)  
    return mu, ub - mu 


