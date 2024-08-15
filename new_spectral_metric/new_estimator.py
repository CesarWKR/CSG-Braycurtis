
import logging
from itertools import product
import numpy as np
import scipy
import scipy.spatial
from numpy.linalg import LinAlgError
from scipy.sparse.csgraph import laplacian
import pickle
from new_lib import find_samples, compute_expectation_with_monte_carlo

log = logging.getLogger(__name__)

class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):
        self.M_sample = M_sample  # Tama�o m�ximo de muestra por clase
        self.k_nearest = k_nearest
        self.distance = distance
        self.P = {}  # Initialize P attribute
        self.M = None  # Initialize M attribute

    def fit(self, data, target):
        np.random.seed(None)
        data_x = data.copy()
        self.n_class = np.max(target) - min(0, np.min(target)) + 1

        # Pasar self.M_sample a find_samples
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

        # Compute the M matrix for samples
        self.M = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                self.M[i, j] = 1 - scipy.spatial.distance.braycurtis(data[i], data[j])

        # for i, j in product(range(self.n_class), range(self.n_class)):
        #     class_i_indices = self.class_indices[i]
        #     class_j_indices = self.class_indices[j]
        #     similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
        #     for idx_i, i_idx in enumerate(class_i_indices):
        #         for idx_j, j_idx in enumerate(class_j_indices):
        #             similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
        #     self.P[(i, j)] = similarity_matrix   # Compute the matriz to compare the pair classes

        # for i, j in product(range(self.n_class), range(self.n_class)):
        #     if i != j:  # Solo comparar clases diferentes
        #         class_i_indices = self.class_indices[i]
        #         class_j_indices = self.class_indices[j]
        #         similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
        #         for idx_i, i_idx in enumerate(class_i_indices):
        #             for idx_j, j_idx in enumerate(class_j_indices):
        #                 similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
        #         self.P[(i, j)] = similarity_matrix   # Almacena la matriz de similitud para clases diferentes


        for i, j in product(range(self.n_class), range(self.n_class)):
            if i < j and i != j:  # Solo comparar clases diferentes y evitar la duplicación
                class_i_indices = self.class_indices[i]
                class_j_indices = self.class_indices[j]
                similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
                for idx_i, i_idx in enumerate(class_i_indices):
                    for idx_j, j_idx in enumerate(class_j_indices):
                        similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
                self.P[(i, j)] = similarity_matrix   # Almacena la matriz de similitud para clases pares
                #self.P[(j, i)] = similarity_matrix.T  # Almacena la matriz transpuesta para la comparación inversa


    def _csg_from_evals(self, evals: np.ndarray) -> float:
        grads = evals[1:] - evals[:-1]
        ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
        csg: float = np.maximum.accumulate(ratios, -1).sum(1)
        return csg

if __name__ == "__main__":
    pass
    