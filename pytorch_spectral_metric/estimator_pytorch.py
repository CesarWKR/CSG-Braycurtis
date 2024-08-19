import logging  
from itertools import product  
import cupy as cp  
import scipy.spatial  
from numpy.linalg import LinAlgError  
from scipy.sparse.csgraph import laplacian  
import torch  
from lib_pytorch import find_samples, compute_expectation_with_monte_carlo  

log = logging.getLogger(__name__)  

class CumulativeGradientEstimator:  
    def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):  
        self.M_sample = M_sample  
        self.k_nearest = k_nearest  
        self.distance = distance  
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'  
        self.P = {}  
        self.C = {}  
        self.M = None  

    def fit(self, data, target):  
        cp.random.seed(None)  
        data_x = cp.array(data)  # Convertir a CuPy para procesamiento en GPU  
        self.n_class = cp.max(target) - min(0, cp.min(target)) + 1  

        class_samples, self.class_indices = find_samples(  
            data_x, cp.array(target), self.n_class, M=self.M_sample  
        )  

        self.compute(data_x, cp.array(target), class_samples)  
        return self  

    def compute(self, data, target, class_samples):  
        self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(  
            data, target, class_samples,  
            class_indices=self.class_indices,  
            n_class=self.n_class,  
            k_nearest=self.k_nearest,  
            distance=self.distance,  
        )  

        self.W = cp.eye(self.n_class)  
        for i, j in product(range(self.n_class), range(self.n_class)):  
            self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(cp.asnumpy(self.S[i]), cp.asnumpy(self.S[j]))  

        self.difference = 1 - self.W  

        self.L_mat, dd = laplacian(cp.asnumpy(self.W), False, True)  
        try:  
            self.evals, self.evecs = cp.linalg.eigh(cp.array(self.L_mat))  
            self.csg = self._csg_from_evals(self.evals)  
        except LinAlgError as e:  
            log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")  
            self.evals = cp.ones([self.n_class]) * cp.nan  
            self.evecs = cp.ones([self.n_class, self.n_class]) * cp.nan  
            self.csg = cp.nan  

        self.M = cp.zeros((data.shape[0], data.shape[0]))  
        for i in range(data.shape[0]):  
            for j in range(data.shape[0]):  
                self.M[i, j] = 1 - scipy.spatial.distance.braycurtis(cp.asnumpy(data[i]), cp.asnumpy(data[j]))  

        for i in range(self.n_class):  
            class_indices = self.class_indices[i]  
            similarity_matrix = cp.zeros((len(class_indices), len(class_indices)))  
            for idx_i, i_idx in enumerate(class_indices):  
                for idx_j, j_idx in enumerate(class_indices):  
                    similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(cp.asnumpy(data[i_idx]), cp.asnumpy(data[j_idx]))  
            self.C[i] = similarity_matrix  

        for i, j in product(range(self.n_class), range(self.n_class)):  
            if i < j and i != j:  
                class_i_indices = self.class_indices[i]  
                class_j_indices = self.class_indices[j]  
                similarity_matrix = cp.zeros((len(class_i_indices), len(class_j_indices)))  
                for idx_i, i_idx in enumerate(class_i_indices):  
                    for idx_j, j_idx in enumerate(class_j_indices):  
                        similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(cp.asnumpy(data[i_idx]), cp.asnumpy(data[j_idx]))  
                self.P[(i, j)] = similarity_matrix  

    def _csg_from_evals(self, evals: cp.ndarray) -> float:  
        grads = evals[1:] - evals[:-1]  
        ratios = grads / (cp.arange(grads.shape[-1], 0, -1) + 1)  
        csg = cp.maximum.accumulate(ratios, -1).sum(1)  
        return csg  

if __name__ == "__main__":  
    pass