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