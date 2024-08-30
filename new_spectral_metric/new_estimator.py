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

#log = logging.getLogger(__name__)

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

 
# log = logging.getLogger(__name__)  

# class SimilarityCalculator:  
#     def __init__(self, device='cuda'):  
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  

#     def compute_similarity_matrix(self, data):  
#         # Convertir los datos a un arreglo de NumPy si es necesario  
#         data_np = np.array(data)  
#         num_samples = data_np.shape[0]  
#         similarity_matrix = np.zeros((num_samples, num_samples))  

#         # Calcular la matriz de similitud usando Bray-Curtis  
#         for i in range(num_samples):  
#             for j in range(num_samples):  
#                 similarity_matrix[i, j] = 1 - scipy.spatial.distance.braycurtis(data_np[i], data_np[j])  

#         return similarity_matrix  

#     def compute_intra_class_similarity(self, data, class_indices):  
#         similarities = {}  
#         for i, indices in class_indices.items():  
#             class_data = data[indices]  
#             similarities[i] = self.compute_similarity_matrix(class_data)  
#         return similarities  

#     def compute_pair_class_similarity(self, data, class_indices):  
#         pair_similarities = {}  
#         for i, indices_i in class_indices.items():  
#             for j, indices_j in class_indices.items():  
#                 if i < j:  
#                     data_i = data[indices_i]  
#                     data_j = data[indices_j]  
#                     num_i = len(data_i)  
#                     num_j = len(data_j)  
#                     similarity_matrix = np.zeros((num_i, num_j))  

#                     # Calcular la matriz de similitud entre pares de clases usando Bray-Curtis  
#                     for idx_i in range(num_i):  
#                         for idx_j in range(num_j):  
#                             similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data_i[idx_i], data_j[idx_j])  

#                     pair_similarities[(i, j)] = similarity_matrix  

#         return pair_similarities


# class CumulativeGradientEstimator(object):  
#     def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
#         self.M_sample = M_sample  
#         self.k_nearest = k_nearest  
#         self.distance = distance  
#         self.P = {}  
#         self.C = {}  
#         self.M = None  
#         self.similarity_calculator = SimilarityCalculator()  # Initialize the similarity calculator  

#     def fit(self, data, target):  
#         np.random.seed(None)  
#         data_x = data.copy()  
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1  

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

#         # Calculate the similarity matrices using the SimilarityCalculator  
#         self.M = self.similarity_calculator.compute_similarity_matrix(data)  
#         self.C = self.similarity_calculator.compute_intra_class_similarity(data, self.class_indices)  
#         self.P = self.similarity_calculator.compute_pair_class_similarity(data, self.class_indices)  

#     def _csg_from_evals(self, evals: np.ndarray) -> float:  
#         grads = evals[1:] - evals[:-1]  
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)  
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)  
#         return csg  

# if __name__ == "__main__":  
#     pass


 

# log = logging.getLogger(__name__)  

# class SimilarityCalculator:  
#     def __init__(self, device='cuda'):  
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  

#     def bray_curtis_similarity(self, data):  
#         data_torch = torch.tensor(data, device=self.device, dtype=torch.float32)  
#         num_samples = data_torch.size(0)  
        
#         # Vectorized computation of Bray-Curtis similarity  
#         data_expanded = data_torch.unsqueeze(1) - data_torch.unsqueeze(0)  
#         abs_diff_sum = torch.sum(torch.abs(data_expanded), dim=-1)  
        
#         data_sum = data_torch.unsqueeze(1) + data_torch.unsqueeze(0)  
#         abs_data_sum = torch.sum(torch.abs(data_sum), dim=-1)  
        
#         similarity_matrix = 1 - (abs_diff_sum / abs_data_sum)  
        
#         return similarity_matrix.cpu().numpy()  

#     def compute_similarity_matrix(self, data):  
#         return self.bray_curtis_similarity(data)  

#     def compute_intra_class_similarity(self, data, class_indices):  
#         similarities = {}  
#         for i, indices in class_indices.items():  
#             class_data = data[indices]  
#             similarities[i] = self.compute_similarity_matrix(class_data)  
#         return similarities  

#     def compute_pair_class_similarity(self, data, class_indices):  
#         pair_similarities = {}  
#         for i, indices_i in class_indices.items():  
#             for j, indices_j in class_indices.items():  
#                 if i < j:  
#                     data_i = data[indices_i]  
#                     data_j = data[indices_j]  
#                     data_i_torch = torch.tensor(data_i, device=self.device, dtype=torch.float32)  
#                     data_j_torch = torch.tensor(data_j, device=self.device, dtype=torch.float32)  
                    
#                     # Vectorized computation for pair class similarity  
#                     data_i_expanded = data_i_torch.unsqueeze(1) - data_j_torch.unsqueeze(0)  
#                     abs_diff_sum = torch.sum(torch.abs(data_i_expanded), dim=-1)  
                    
#                     data_sum = data_i_torch.unsqueeze(1) + data_j_torch.unsqueeze(0)  
#                     abs_data_sum = torch.sum(torch.abs(data_sum), dim=-1)  
                    
#                     similarity_matrix = 1 - (abs_diff_sum / abs_data_sum)  
                    
#                     pair_similarities[(i, j)] = similarity_matrix.cpu().numpy()  

#         return pair_similarities  

# class CumulativeGradientEstimator(object):  
#     def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
#         self.M_sample = M_sample  
#         self.k_nearest = k_nearest  
#         self.distance = distance  
#         self.P = {}  
#         self.C = {}  
#         self.M = None  
#         self.similarity_calculator = SimilarityCalculator()  # Initialize the similarity calculator  

#     def fit(self, data, target):  
#         np.random.seed(None)  
#         data_x = data.copy()  
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1  

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
#             log.warning(f"{str(e)}; assigning evals,evecs,csg to NaN")  
#             self.evals = np.ones([self.n_class]) * np.nan  
#             self.evecs = np.ones([self.n_class, self.n_class]) * np.nan  
#             self.csg = np.nan  

#         # Calculate the similarity matrices using the SimilarityCalculator  
#         self.M = self.similarity_calculator.compute_similarity_matrix(data)  
#         self.C = self.similarity_calculator.compute_intra_class_similarity(data, self.class_indices)  
#         self.P = self.similarity_calculator.compute_pair_class_similarity(data, self.class_indices)  

#     def _csg_from_evals(self, evals: np.ndarray) -> float:  
#         grads = evals[1:] - evals[:-1]  
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)  
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)  
#         return csg  

# if __name__ == "__main__":  
#     pass


# log = logging.getLogger(__name__)  

# class SimilarityCalculator:  
#     def __init__(self, device='cuda'):  
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  

#     def bray_curtis_similarity_batch(self, data, batch_size=100):  
#         data_torch = torch.tensor(data, device=self.device, dtype=torch.float32)  
#         num_samples = data_torch.size(0)  
#         similarity_matrix = torch.zeros((num_samples, num_samples), device=self.device)  

#         for start_idx in range(0, num_samples, batch_size):  
#             end_idx = min(start_idx + batch_size, num_samples)  
#             batch_data = data_torch[start_idx:end_idx]  

#             # Calculate diff and summation for the current batch  
#             data_diff = batch_data.unsqueeze(1) - data_torch  
#             abs_diff_sum = torch.sum(torch.abs(data_diff), dim=-1)  

#             data_sum = batch_data.unsqueeze(1) + data_torch  
#             abs_data_sum = torch.sum(torch.abs(data_sum), dim=-1)  

#             similarity_batch = 1 - (abs_diff_sum / abs_data_sum)  
#             similarity_matrix[start_idx:end_idx] = similarity_batch  

#         return similarity_matrix.cpu().numpy()  

#     def compute_similarity_matrix(self, data, batch_size=500):  
#         return self.bray_curtis_similarity_batch(data, batch_size=batch_size)  

#     def compute_intra_class_similarity(self, data, class_indices, batch_size=100):  
#         similarities = {}  
#         for i, indices in class_indices.items():  
#             class_data = data[indices]  
#             similarities[i] = self.compute_similarity_matrix(class_data, batch_size=batch_size)  
#         return similarities  

#     def compute_pair_class_similarity(self, data, class_indices, batch_size=100):  
#         pair_similarities = {}  
#         for i, indices_i in class_indices.items():  
#             for j, indices_j in class_indices.items():  
#                 if i < j:  
#                     data_i = data[indices_i]  
#                     data_j = data[indices_j]  
#                     data_i_torch = torch.tensor(data_i, device=self.device, dtype=torch.float32)  
#                     data_j_torch = torch.tensor(data_j, device=self.device, dtype=torch.float32)  

#                     similarity_matrix = torch.zeros((data_i_torch.size(0), data_j_torch.size(0)), device=self.device)  

#                     for start_idx in range(0, data_i_torch.size(0), batch_size):  
#                         end_idx = min(start_idx + batch_size, data_i_torch.size(0))  
#                         batch_data_i = data_i_torch[start_idx:end_idx]  

#                         data_i_diff = batch_data_i.unsqueeze(1) - data_j_torch  
#                         abs_diff_sum = torch.sum(torch.abs(data_i_diff), dim=-1)  

#                         data_i_sum = batch_data_i.unsqueeze(1) + data_j_torch  
#                         abs_data_sum = torch.sum(torch.abs(data_i_sum), dim=-1)  

#                         similarity_batch = 1 - (abs_diff_sum / abs_data_sum)  
#                         similarity_matrix[start_idx:end_idx] = similarity_batch  

#                     pair_similarities[(i, j)] = similarity_matrix.cpu().numpy()  

#         return pair_similarities  

# class CumulativeGradientEstimator:  
#     def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
#         self.M_sample = M_sample  
#         self.k_nearest = k_nearest  
#         self.distance = distance  
#         self.P = {}  
#         self.C = {}  
#         self.M = None  
#         self.similarity_calculator = SimilarityCalculator()  # Initialize the similarity calculator  

#     def fit(self, data, target):  
#         np.random.seed(None)  
#         data_x = data.copy()  
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1  

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
#             log.warning(f"{str(e)}; assigning evals,evecs,csg to NaN")  
#             self.evals = np.ones([self.n_class]) * np.nan  
#             self.evecs = np.ones([self.n_class, self.n_class]) * np.nan  
#             self.csg = np.nan  

#         # Calculate the similarity matrices using the SimilarityCalculator  
#         self.M = self.similarity_calculator.compute_similarity_matrix(data)  
#         self.C = self.similarity_calculator.compute_intra_class_similarity(data, self.class_indices)  
#         self.P = self.similarity_calculator.compute_pair_class_similarity(data, self.class_indices)  

#     def _csg_from_evals(self, evals: np.ndarray) -> float:  
#         grads = evals[1:] - evals[:-1]  
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)  
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)  
#         return csg  

# if __name__ == "__main__":  
#     pass

import logging  
import torch  
import numpy as np  
from itertools import product  
from scipy.linalg import LinAlgError  

log = logging.getLogger(__name__)  

class SimilarityCalculator:  
    def __init__(self, device='cuda'):  
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')  

    def bray_curtis_similarity_blocked(self, data, block_size=100):  
        data_torch = torch.tensor(data, device=self.device, dtype=torch.float16)  
        num_samples = data_torch.size(0)  
        similarity_matrix = torch.zeros((num_samples, num_samples), device='cpu', dtype=torch.float16)  
        
        # Create blocks for both dimensions  
        for start_i in range(0, num_samples, block_size):  
            end_i = min(start_i + block_size, num_samples)  

            for start_j in range(0, num_samples, block_size):  
                end_j = min(start_j + block_size, num_samples)  

                block_i = data_torch[start_i:end_i]  
                block_j = data_torch[start_j:end_j]  
                
                # Compute differences and sums for blocks  
                diff = block_i.unsqueeze(1) - block_j  
                diff_sum = torch.sum(torch.abs(diff), dim=-1)  
                
                summation = block_i.unsqueeze(1) + block_j  
                abs_sum = torch.sum(torch.abs(summation), dim=-1)  
                
                similarity_block = 1 - (diff_sum / abs_sum)  
                similarity_matrix[start_i:end_i, start_j:end_j] = similarity_block.cpu()  
                
        return similarity_matrix.numpy()  

    def compute_similarity_matrix(self, data, block_size=100):  
        return self.bray_curtis_similarity_blocked(data, block_size=block_size)  

    def compute_intra_class_similarity(self, data, class_indices, block_size=100):  
        similarities = {}  
        for i, indices in class_indices.items():  
            class_data = data[indices]  
            similarities[i] = self.compute_similarity_matrix(class_data, block_size=block_size)  
        return similarities  

    def compute_pair_class_similarity(self, data, class_indices, block_size=100):  
        pair_similarities = {}  
        for i, indices_i in class_indices.items():  
            for j, indices_j in class_indices.items():  
                if i < j:  
                    data_i = data[indices_i]  
                    data_j = data[indices_j]  
                    data_i_torch = torch.tensor(data_i, device=self.device, dtype=torch.float16)  
                    data_j_torch = torch.tensor(data_j, device=self.device, dtype=torch.float16)  

                    similarity_matrix = torch.zeros((data_i_torch.size(0), data_j_torch.size(0)), device='cpu', dtype=torch.float16)  

                    for start_idx in range(0, data_i_torch.size(0), block_size):  
                        end_idx = min(start_idx + block_size, data_i_torch.size(0))  
                        batch_data_i = data_i_torch[start_idx:end_idx]  

                        data_i_diff = batch_data_i.unsqueeze(1) - data_j_torch  
                        abs_diff_sum = torch.sum(torch.abs(data_i_diff), dim=-1)  

                        data_i_sum = batch_data_i.unsqueeze(1) + data_j_torch  
                        abs_data_sum = torch.sum(torch.abs(data_i_sum), dim=-1)  

                        similarity_batch = 1 - (abs_diff_sum / abs_data_sum)  
                        similarity_matrix[start_idx:end_idx] = similarity_batch.cpu()  

                    pair_similarities[(i, j)] = similarity_matrix.numpy()  

        return pair_similarities  

class CumulativeGradientEstimator:  
    def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
        self.M_sample = M_sample  
        self.k_nearest = k_nearest  
        self.distance = distance  
        self.P = {}  
        self.C = {}  
        self.M = None  
        self.similarity_calculator = SimilarityCalculator()  # Initialize similarity calculator  

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
            log.warning(f"{str(e)}; assigning evals,evecs,csg to NaN")  
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