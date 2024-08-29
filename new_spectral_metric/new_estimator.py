import logging
from itertools import product
import torch  
import cupy
import numpy as np
import scipy
import scipy.spatial
from numpy.linalg import LinAlgError
from scipy.sparse.csgraph import laplacian
import pickle
from new_lib import find_samples, compute_expectation_with_monte_carlo

log = logging.getLogger(__name__)

# class CumulativeGradientEstimator(object):
#     def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):
#         self.M_sample = M_sample  # Tamaño máximo de muestra por clase
#         self.k_nearest = k_nearest
#         self.distance = distance
#         self.P = {}  # Initialize P attribute
#         self.C = {}  # Initialize C attribute for intra-class similarity
#         self.M = None  # Initialize M attribute

#     def fit(self, data, target):
#         np.random.seed(None)
#         data_x = data.copy()
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1

#         # Pasar self.M_sample a find_samples
#         class_samples, self.class_indices = find_samples(
#             data_x, target, self.n_class, M=self.M_sample
#         )

#         self.compute(data_x, target, class_samples)
#         return self

#     def compute(self, data, target, class_samples):
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )

#         self.W = np.eye(self.n_class)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

#         self.difference = 1 - self.W

#         self.L_mat, dd = laplacian(self.W, False, True)
#         try:
#             self.evals, self.evecs = np.linalg.eigh(self.L_mat)
#             self.csg = self._csg_from_evals(self.evals)
#         except LinAlgError as e:
#             log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")
#             self.evals = np.ones([self.n_class]) * np.nan
#             self.evecs = np.ones([self.n_class, self.n_class]) * np.nan
#             self.csg = np.nan

#         # Compute the M matrix for all samples
#         self.M = np.zeros((data.shape[0], data.shape[0]))
#         for i in range(data.shape[0]):
#             for j in range(data.shape[0]):
#                 self.M[i, j] = 1 - scipy.spatial.distance.braycurtis(data[i], data[j])

#         # Compute the C matrix for intra-class similarity
#         for i in range(self.n_class):
#             class_indices = self.class_indices[i]
#             similarity_matrix = np.zeros((len(class_indices), len(class_indices)))
#             for idx_i, i_idx in enumerate(class_indices):
#                 for idx_j, j_idx in enumerate(class_indices):
#                     similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
#             self.C[i] = similarity_matrix   # Store the similarity matrix for each class

#         for i, j in product(range(self.n_class), range(self.n_class)):
#             if i < j and i != j:  # Solo comparar clases diferentes y evitar la duplicación
#                 class_i_indices = self.class_indices[i]
#                 class_j_indices = self.class_indices[j]
#                 similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
#                 for idx_i, i_idx in enumerate(class_i_indices):
#                     for idx_j, j_idx in enumerate(class_j_indices):
#                         similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
#                 self.P[(i, j)] = similarity_matrix   # Almacena la matriz de similitud para clases pares

#     def _csg_from_evals(self, evals: np.ndarray) -> float:
#         grads = evals[1:] - evals[:-1]
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)
#         return csg

# if __name__ == "__main__":
#     pass

 
log = logging.getLogger(__name__)  

class SimilarityCalculator:  
    def __init__(self, device='cuda'):  
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  

    def compute_similarity_matrix(self, data):  
        data_torch = torch.tensor(data).to(self.device)  
        diffs = data_torch[:, None, :] - data_torch[None, :, :]  
        norms = torch.norm(diffs, dim=-1)  
        max_norm = norms.max()  
        similarity_matrix = 1 - (norms / max_norm)  
        return similarity_matrix.cpu().numpy()  

    def compute_intra_class_similarity(self, data, class_indices):  
        similarities = {}  
        for i, indices in class_indices.items():  
            class_data = data[indices]  
            similarities[i] = self.compute_similarity_matrix(class_data)  
        return similarities  

    def compute_pair_class_similarity(self, data, class_indices):  
        pair_similarities = {}  
        for i, indices_i in class_indices.items():  
            for j, indices_j in class_indices.items():  
                if i < j:  
                    data_i = data[indices_i]  
                    data_j = data[indices_j]  
                    data_i_torch = torch.tensor(data_i).to(self.device)  
                    data_j_torch = torch.tensor(data_j).to(self.device)  
                    pair_diff = data_i_torch[:, None, :] - data_j_torch[None, :, :]  
                    pair_norms = torch.norm(pair_diff, dim=-1)  
                    max_pair_norm = pair_norms.max()  
                    similarity_matrix = 1 - (pair_norms / max_pair_norm)  
                    pair_similarities[(i, j)] = similarity_matrix.cpu().numpy()  
        return pair_similarities  


class CumulativeGradientEstimator(object):  
    def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
        self.M_sample = M_sample  
        self.k_nearest = k_nearest  
        self.distance = distance  
        self.P = {}  
        self.C = {}  
        self.M = None  
        self.similarity_calculator = SimilarityCalculator()  # Initialize the similarity calculator  

    def fit(self, data, target):  
        np.random.seed(None)  
        data_x = data.copy()  
        self.n_class = np.max(target) - min(0, np.min(target)) + 1  

        class_samples, self.class_indices = find_samples(  
            data_x, target, self.n_class, M=self.M_sample  
        )  

        self.compute(data_x, target, class_samples)  
        return self  

    def compute(self, data, target, class_samples):  
        self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(  
            data,  
            target,  
            class_samples,  
            class_indices=self.class_indices,  
            n_class=self.n_class,  
            k_nearest=self.k_nearest,  
            distance=self.distance,  
        )  

        self.W = np.eye(self.n_class)  
        for i, j in product(range(self.n_class), range(self.n_class)):  
            self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])  

        self.difference = 1 - self.W  

        self.L_mat, dd = laplacian(self.W, False, True)  
        try:  
            self.evals, self.evecs = np.linalg.eigh(self.L_mat)  
            self.csg = self._csg_from_evals(self.evals)  
        except LinAlgError as e:  
            log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")  
            self.evals = np.ones([self.n_class]) * np.nan  
            self.evecs = np.ones([self.n_class, self.n_class]) * np.nan  
            self.csg = np.nan  

        # Calculate the similarity matrices using the SimilarityCalculator  
        self.M = self.similarity_calculator.compute_similarity_matrix(data)  
        self.C = self.similarity_calculator.compute_intra_class_similarity(data, self.class_indices)  
        self.P = self.similarity_calculator.compute_pair_class_similarity(data, self.class_indices)  

    def _csg_from_evals(self, evals: np.ndarray) -> float:  
        grads = evals[1:] - evals[:-1]  
        ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)  
        csg: float = np.maximum.accumulate(ratios, -1).sum(1)  
        return csg  

if __name__ == "__main__":  
    pass