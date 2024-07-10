"""

import logging
from itertools import product

import numpy as np
import scipy
import scipy.spatial
from numpy.linalg import LinAlgError
from scipy.sparse.csgraph import laplacian

from spectral_metric.lib import find_samples, compute_expectation_with_monte_carlo

log = logging.getLogger(__name__)


class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=250, k_nearest=3, distance="euclidean"):
        # 
        # The Cumulative Gradient Estimator, estimates the complexity of a dataset.
        # Args:
        #     M_sample (int): Number of sample per class to use
        #     k_nearest (int): Number of neighbours to look to compute $P(C_c \vert x)$.
        #     distance: name of the distance to use.
        # 
        self.M_sample = M_sample
        self.k_nearest = k_nearest
        self.distance = distance

    def fit(self, data, target):
        # 
        # Estimate the CSG metric from the data
        # Args:
        #     data: data samples, ndarray (n_samples, n_features)
        #     target: target samples, ndarray (n_samples)
        # 
        np.random.seed(None)
        data_x = data.copy()
        self.n_class = np.max(target) - min(0, np.min(target)) + 1

        # Do class sampling
        class_samples, self.class_indices = find_samples(
            data_x, target, self.n_class, M=self.M_sample
        )

        self.compute(data_x, target, class_samples)
        return self

    def compute(self, data, target, class_samples):
        # 
        # Compute the difference matrix and the eigenvalues
        # Args:
        #     data: data samples, ndarray (n_samples, n_features)
        #     target: target samples, ndarray (n_samples)
        #     class_samples : class samples, Dict[class_idx, Array[M, n_features]]
        # 
        # Compute E_{p(x\mid C_i)} [p(x\mid C_j)]
        self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
            data,
            target,
            class_samples,
            class_indices=self.class_indices,
            n_class=self.n_class,
            k_nearest=self.k_nearest,
            distance=self.distance,
        )

        # Compute the D matrix
        self.W = np.eye(self.n_class)
        for i, j in product(range(self.n_class), range(self.n_class)):
            self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

        self.difference = 1 - self.W

        # Get the Laplacian and its eigen values
        self.L_mat, dd = laplacian(self.W, False, True)
        try:
            self.evals, self.evecs = np.linalg.eigh(self.L_mat)
            self.csg = self._csg_from_evals(self.evals)
        except LinAlgError as e:
            log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")
            self.evals = np.ones([self.n_class]) * np.nan
            self.evecs = np.ones([self.n_class, self.n_class]) * np.nan
            self.csg = np.nan

    def _csg_from_evals(self, evals: np.ndarray) -> float:
        # [n_class]
        grads = evals[1:] - evals[:-1]
        ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
        csg: float = np.maximum.accumulate(ratios, -1).sum(1)
        return csg
"""


import logging
from itertools import product

import numpy as np
import scipy
import scipy.spatial
from numpy.linalg import LinAlgError
from scipy.sparse.csgraph import laplacian

from spectral_metric.lib import find_samples, compute_expectation_with_monte_carlo


log = logging.getLogger(__name__)


class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=5000, k_nearest=10, distance="euclidean"):
        self.M_sample = M_sample
        self.k_nearest = k_nearest
        self.distance = distance
        self.P = {}  # Initialize P attribute
        self.M = None  # Initialize M attribute

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

        self.M = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                self.M[i, j] = 1 - scipy.spatial.distance.braycurtis(data[i], data[j])

        for i, j in product(range(self.n_class), range(self.n_class)):
            class_i_indices = self.class_indices[i]
            class_j_indices = self.class_indices[j]
            similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
            for idx_i, i_idx in enumerate(class_i_indices):
                for idx_j, j_idx in enumerate(class_j_indices):
                    similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
            self.P[(i, j)] = similarity_matrix

    def _csg_from_evals(self, evals: np.ndarray) -> float:
        grads = evals[1:] - evals[:-1]
        ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1] + 1)))]) + 1)
        csg: float = np.maximum.accumulate(ratios, -1).sum(1)
        return csg

if __name__ == "__main__":
    # Datos de ejemplo
    from sklearn.datasets import load_iris
    data = load_iris()
    X = data.data
    y = data.target

    # Inicializar y ajustar el estimador
    estimator = CumulativeGradientEstimator(M_sample=5000, k_nearest=10, distance="euclidean")
    estimator.fit(X, y)

    # Imprimir resultados
    log.info(f"Evals: {estimator.evals}")
    log.info(f"Evecs: {estimator.evecs}")
    log.info(f"CSG: {estimator.csg}")


import matplotlib.pyplot as plt

def plot_eigenvalues(evals):
    plt.figure(figsize=(10, 6))
    plt.plot(evals, 'bo-', label="Eigenvalues")
    plt.title("Eigenvalues of the Laplacian")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid(True)
    plt.show()

# Llama a la función de visualización
plot_eigenvalues(estimator.evals)

print("\n")
# Verificar la existencia del atributo P
if hasattr(estimator, 'P'):
    print("El atributo P existe.")
else:
    print("El atributo P no existe.")

# Verificar la existencia del atributo M
if estimator.M is not None:
    print("El atributo M existe.")
else:
    print("El atributo M no existe.")

# Asegurarse de que P esté inicializado correctamente
print(f"Valor de P: {estimator.P}")

import pickle

with open('estimator.pkl', 'wb') as f:
    pickle.dump(estimator, f)

print("El objeto estimator ha sido guardado en 'estimator.pkl'.")


# import logging
# from itertools import product

# import numpy as np
# import scipy
# import scipy.spatial
# from numpy.linalg import LinAlgError
# from scipy.sparse.csgraph import laplacian

# from spectral_metric.lib import find_samples, compute_expectation_with_monte_carlo

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

# class CumulativeGradientEstimator(object):
#     def __init__(self, M_sample=250, k_nearest=3, distance="euclidean"):
#         """
#         The Cumulative Gradient Estimator, estimates the complexity of a dataset.
#         Args:
#             M_sample (int): Number of sample per class to use
#             k_nearest (int): Number of neighbours to look to compute $P(C_c \vert x)$.
#             distance: name of the distance to use.
#         """
#         self.M_sample = M_sample
#         self.k_nearest = k_nearest
#         self.distance = distance

#     def fit(self, data, target):
#         """
#         Estimate the CSG metric from the data
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#         """
#         np.random.seed(None)
#         data_x = data.copy()
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1

#         # Do class sampling
#         class_samples, self.class_indices = find_samples(
#             data_x, target, self.n_class, M=self.M_sample
#         )

#         self.compute(data_x, target, class_samples)
#         return self

#     def compute(self, data, target, class_samples):
#         """
#         Compute the difference matrix and the eigenvalues
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#             class_samples : class samples, Dict[class_idx, Array[M, n_features]]
#         """
#         # Compute E_{p(x\mid C_i)} [p(x\mid C_j)]
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )

#         # Compute the D matrix for classes
#         self.W = np.eye(self.n_class)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

#         self.difference = 1 - self.W

#         # Get the Laplacian and its eigen values
#         self.L_mat, dd = laplacian(self.W, False, True)
#         try:
#             self.evals, self.evecs = np.linalg.eigh(self.L_mat)
#             self.csg = self._csg_from_evals(self.evals)
#         except LinAlgError as e:
#             log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")
#             self.evals = np.ones([self.n_class]) * np.nan
#             self.evecs = np.ones([self.n_class, self.n_class]) * np.nan
#             self.csg = np.nan

#         # Compute the M matrix for samples
#         self.M = np.zeros((data.shape[0], data.shape[0]))
#         for i in range(data.shape[0]):
#             for j in range(data.shape[0]):
#                 self.M[i, j] = 1 - scipy.spatial.distance.braycurtis(data[i], data[j])

#     def _csg_from_evals(self, evals: np.ndarray) -> float:
#         # [n_class]
#         grads = evals[1:] - evals[:-1]
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1 + 1])))]) + 1)
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)
#         return csg

# if __name__ == "__main__":
#     # Datos de ejemplo
#     from sklearn.datasets import load_iris
#     data = load_iris()
#     X = data.data
#     y = data.target

#     # Inicializar y ajustar el estimador
#     estimator = CumulativeGradientEstimator(M_sample=50, k_nearest=5, distance="euclidean")
#     estimator.fit(X, y)

#     # Imprimir resultados
#     log.info(f"Evals: {estimator.evals}")
#     log.info(f"Evecs: {estimator.evecs}")
#     log.info(f"CSG: {estimator.csg}")


# import matplotlib.pyplot as plt

# def plot_eigenvalues(evals):
#     plt.figure(figsize=(10, 6))
#     plt.plot(evals, 'bo-', label="Eigenvalues")
#     plt.title("Eigenvalues of the Laplacian")
#     plt.xlabel("Index")
#     plt.ylabel("Eigenvalue")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Llama a la función de visualización
# plot_eigenvalues(estimator.evals)

##############################################################################################################################################

# import numpy as np
# import scipy
# from scipy.spatial.distance import braycurtis
# from scipy.sparse.csgraph import laplacian
# from numpy.linalg import LinAlgError
# from itertools import product
# from typing import Dict, List
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
# import logging 

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

# def find_samples(data_x, target, n_class, M):
#     class_samples = {}
#     class_indices = {}
#     for c in range(n_class):
#         indices = np.where(target == c)[0]
#         class_indices[c] = indices
#         if len(indices) > M:
#             sampled_indices = np.random.choice(indices, M, replace=False)
#         else:
#             sampled_indices = indices
#         class_samples[c] = data_x[sampled_indices]
#     return class_samples, class_indices

# def compute_expectation_with_monte_carlo(data, target, class_samples, class_indices, n_class, k_nearest, distance):
#     S = np.zeros((n_class, data.shape[1]))
#     similarity_arrays = {}
#     for c in range(n_class):
#         samples = class_samples[c]
#         similarities = []
#         for i in range(samples.shape[0]):
#             dists = np.linalg.norm(data - samples[i], axis=1)
#             nearest_indices = np.argsort(dists)[:k_nearest]
#             similarities.append(np.mean(data[nearest_indices], axis=0))
#         similarity_arrays[c] = similarities
#         S[c] = np.mean(similarities, axis=0)
#     return S, similarity_arrays

# def laplacian(W, normed, return_diag):
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     return (L, D)

# class CumulativeGradientEstimator:
#     def __init__(self, M_sample=5000, k_nearest=10, distance="euclidean"):
#         """
#         The Cumulative Gradient Estimator estimates the complexity of a dataset.
        
#         Args:
#             M_sample (int): Number of samples per class to use
#             k_nearest (int): Number of neighbours to look to compute $P(C_c \mid x)$.
#             distance (str): Name of the distance to use.
#         """
#         self.M_sample = M_sample
#         self.k_nearest = k_nearest
#         self.distance = distance
#         self.P = {}  # Initialize P attribute
#         self.M = None  # Initialize M attribute

#     def fit(self, data, target):
#         """
#         Estimate the CSG metric from the data.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#         """
#         np.random.seed(None)
#         data_x = data.copy()
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1

#         # Do class sampling
#         class_samples, self.class_indices = find_samples(
#             data_x, target, self.n_class, M=self.M_sample
#         )

#         self.compute(data_x, target, class_samples)
#         return self

#     def compute(self, data, target, class_samples):
#         """
#         Compute the difference matrix and the eigenvalues.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#             class_samples: class samples, Dict[class_idx, Array[M, n_features]]
#         """
#         # Compute E_{p(x|C_i)} [p(x|C_j)]
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )

#         # Compute the D matrix for classes
#         self.W = np.eye(self.n_class)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

#         self.difference = 1 - self.W

#         # Get the Laplacian and its eigenvalues
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

#         # Compute the P matrix for each pair of classes
#         self.P = {}
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             class_i_indices = self.class_indices[i]
#             class_j_indices = self.class_indices[j]
#             similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
#             for idx_i, i_idx in enumerate(class_i_indices):
#                 for idx_j, j_idx in enumerate(class_j_indices):
#                     similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
#             self.P[(i, j)] = similarity_matrix

#     def _csg_from_evals(self, evals: np.ndarray) -> float:
#         # [n_class]
#         grads = evals[1:] - evals[:-1]
#         ratios = grads / (np.array([list(reversed(range(1, grads.shape[-1 + 1])))]) + 1)
#         csg: float = np.maximum.accumulate(ratios, -1).sum(1)
#         return csg

# # Datos de ejemplo
# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#     data = load_iris()
#     X = data.data
#     y = data.target

#     # Inicializar y ajustar el estimador
#     estimator = CumulativeGradientEstimator(M_sample=5000, k_nearest=10, distance="euclidean")
#     estimator.fit(X, y)

#     # Imprimir resultados
#     log.info(f"Evals: {estimator.evals}")
#     log.info(f"Evecs: {estimator.evecs}")
#     log.info(f"CSG: {estimator.csg}")

# import matplotlib.pyplot as plt

# def plot_eigenvalues(evals):
#     plt.figure(figsize=(10, 6))
#     plt.plot(evals, 'bo-', label="Eigenvalues")
#     plt.title("Eigenvalues of the Laplacian")
#     plt.xlabel("Index")
#     plt.ylabel("Eigenvalue")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Llama a la función de visualización
# plot_eigenvalues(estimator.evals)

# print("\n")
# # Verificar la existencia del atributo P
# if hasattr(estimator, 'P'):
#     print("El atributo P existe.")
# else:
#     print("El atributo P no existe.")

# # Verificar la existencia del atributo M
# if estimator.M is not None:
#     print("El atributo M existe.")
# else:
#     print("El atributo M no existe.")

# # Asegurarse de que P esté inicializado correctamente
# print(f"Valor de P: {estimator.P}")

# import pickle

# with open('estimator.pkl', 'wb') as f:
#     pickle.dump(estimator, f)

# print("El objeto estimator ha sido guardado en 'estimator.pkl'.")


#####################################################################################################################################


# import numpy as np
# import scipy
# from scipy.spatial.distance import braycurtis
# from scipy.sparse.csgraph import laplacian
# from numpy.linalg import LinAlgError
# from itertools import product
# from typing import Dict, List
# import logging
# import pathlib

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

# def find_samples(data_x, target, n_class, M):
#     class_samples = {}
#     class_indices = {}
#     for c in range(n_class):
#         indices = np.where(target == c)[0]
#         class_indices[c] = indices
#         if len(indices) > M:
#             sampled_indices = np.random.choice(indices, M, replace=False)
#         else:
#             sampled_indices = indices
#         class_samples[c] = data_x[sampled_indices]
#     return class_samples, class_indices

# def compute_expectation_with_monte_carlo(data, target, class_samples, class_indices, n_class, k_nearest, distance):
#     S = np.zeros((n_class, data.shape[1]))
#     similarity_arrays = {}
#     for c in range(n_class):
#         samples = class_samples[c]
#         similarities = []
#         for i in range(samples.shape[0]):
#             dists = np.linalg.norm(data - samples[i], axis=1)
#             nearest_indices = np.argsort(dists)[:k_nearest]
#             similarities.append(np.mean(data[nearest_indices], axis=0))
#         similarity_arrays[c] = similarities
#         S[c] = np.mean(similarities, axis=0)
#     return S, similarity_arrays

# def laplacian(W, normed, return_diag):
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     return (L, D)

# class CumulativeGradientEstimator:
#     def __init__(self, M_sample=5000, k_nearest=3, distance="euclidean"):
#         """
#         The Cumulative Gradient Estimator estimates the complexity of a dataset.
        
#         Args:
#             M_sample (int): Number of samples per class to use
#             k_nearest (int): Number of neighbours to look to compute $P(C_c \mid x)$.
#             distance (str): Name of the distance to use.
#         """
#         self.M_sample = M_sample
#         self.k_nearest = k_nearest
#         self.distance = distance
#         self.P = {}  # Initialize P attribute
#         self.M = None  # Initialize M attribute

#     def fit(self, data, target):
#         """
#         Estimate the CSG metric from the data.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#         """
#         np.random.seed(None)
#         data_x = data.copy()
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1

#         # Do class sampling
#         class_samples, self.class_indices = find_samples(
#             data_x, target, self.n_class, M=self.M_sample
#         )

#         self.compute(data_x, target, class_samples)
#         return self

#     def compute(self, data, target, class_samples):
#         """
#         Compute the difference matrix and the eigenvalues.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#             class_samples: class samples, Dict[class_idx, Array[M, n_features]]
#         """
#         # Compute E_{p(x|C_i)} [p(x|C_j)]
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )

#         # Compute the D matrix for classes
#         self.W = np.eye(self.n_class)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

#         self.difference = 1 - self.W

#         # Get the Laplacian and its eigenvalues
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

#         # Compute the P matrix for each pair of classes
#         self.P = {}
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             class_i_indices = self.class_indices[i]
#             class_j_indices = self.class_indices[j]
#             similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
#             for idx_i, i_idx in enumerate(class_i_indices):
#                 for idx_j, j_idx in enumerate(class_j_indices):
#                     similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
#             self.P[(i, j)] = similarity_matrix

#     def _csg_from_evals(self, evals: np.ndarray) -> float:
#         # [n_class]
#         grads = evals[1:] - evals[:-1]
#         ratios = grads / (np.array(list(reversed(range(1, grads.shape[0] + 1)))) + 1)
#         csg: float = np.maximum.accumulate(ratios, -1).sum(0)
#         return csg

# # Datos de ejemplo
# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#     data = load_iris()
#     X = data.data
#     y = data.target

#     # Inicializar y ajustar el estimador
#     estimator = CumulativeGradientEstimator(M_sample=5000, k_nearest=5, distance="euclidean")
#     estimator.fit(X, y)

#     # Imprimir resultados
#     log.info(f"Evals: {estimator.evals}")
#     log.info(f"Evecs: {estimator.evecs}")
#     log.info(f"CSG: {estimator.csg}")

# import matplotlib.pyplot as plt

# def plot_eigenvalues(evals):
#     plt.figure(figsize=(10, 6))
#     plt.plot(evals, 'bo-', label="Eigenvalues")
#     plt.title("Eigenvalues of the Laplacian")
#     plt.xlabel("Index")
#     plt.ylabel("Eigenvalue")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Llama a la función de visualización
# plot_eigenvalues(estimator.evals)


# import pickle

# with open('estimator.pkl', 'wb') as f:
#     pickle.dump(estimator, f)

# print("El objeto estimator ha sido guardado en 'estimator.pkl'.")

#######################################################################################################################################

# import numpy as np
# import scipy
# from scipy.spatial.distance import braycurtis
# from scipy.sparse.csgraph import laplacian
# from numpy.linalg import LinAlgError
# from itertools import product
# from typing import Dict, List
# import logging
# import pathlib

# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)

# def find_samples(data_x, target, n_class, M):
#     class_samples = {}
#     class_indices = {}
#     for c in range(n_class):
#         indices = np.where(target == c)[0]
#         class_indices[c] = indices
#         if len(indices) > M:
#             sampled_indices = np.random.choice(indices, M, replace=False)
#         else:
#             sampled_indices = indices
#         class_samples[c] = data_x[sampled_indices]
#     return class_samples, class_indices

# def compute_expectation_with_monte_carlo(data, target, class_samples, class_indices, n_class, k_nearest, distance):
#     S = np.zeros((n_class, data.shape[1]))
#     similarity_arrays = {}
#     for c in range(n_class):
#         samples = class_samples[c]
#         similarities = []
#         for i in range(samples.shape[0]):
#             dists = np.linalg.norm(data - samples[i], axis=1)
#             nearest_indices = np.argsort(dists)[:k_nearest]
#             similarities.append(np.mean(data[nearest_indices], axis=0))
#         similarity_arrays[c] = similarities
#         S[c] = np.mean(similarities, axis=0)
#     return S, similarity_arrays

# def laplacian(W, normed, return_diag):
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     return (L, D)

# class CumulativeGradientEstimator:
#     def __init__(self, M_sample=5000, k_nearest=3, distance="euclidean"):
#         """
#         The Cumulative Gradient Estimator estimates the complexity of a dataset.
        
#         Args:
#             M_sample (int): Number of samples per class to use
#             k_nearest (int): Number of neighbours to look to compute $P(C_c \mid x)$.
#             distance (str): Name of the distance to use.
#         """
#         self.M_sample = M_sample
#         self.k_nearest = k_nearest
#         self.distance = distance
#         self.P = {}  # Initialize P attribute
#         self.M = None  # Initialize M attribute

#     def fit(self, data, target):
#         """
#         Estimate the CSG metric from the data.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#         """
#         np.random.seed(None)
#         data_x = data.copy()
#         self.n_class = np.max(target) - min(0, np.min(target)) + 1

#         # Do class sampling
#         class_samples, self.class_indices = find_samples(
#             data_x, target, self.n_class, M=self.M_sample
#         )

#         self.compute(data_x, target, class_samples)
#         return self

#     def compute(self, data, target, class_samples):
#         """
#         Compute the difference matrix and the eigenvalues.
        
#         Args:
#             data: data samples, ndarray (n_samples, n_features)
#             target: target samples, ndarray (n_samples)
#             class_samples: class samples, Dict[class_idx, Array[M, n_features]]
#         """
#         # Compute E_{p(x|C_i)} [p(x|C_j)]
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )

#         # Compute the D matrix for classes
#         self.W = np.eye(self.n_class)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - scipy.spatial.distance.braycurtis(self.S[i], self.S[j])

#         self.difference = 1 - self.W

#         # Get the Laplacian and its eigenvalues
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

#         # Compute the P matrix for each pair of classes
#         self.P = {}
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             class_i_indices = self.class_indices[i]
#             class_j_indices = self.class_indices[j]
#             similarity_matrix = np.zeros((len(class_i_indices), len(class_j_indices)))
#             for idx_i, i_idx in enumerate(class_i_indices):
#                 for idx_j, j_idx in enumerate(class_j_indices):
#                     similarity_matrix[idx_i, idx_j] = 1 - scipy.spatial.distance.braycurtis(data[i_idx], data[j_idx])
#             self.P[(i, j)] = similarity_matrix

#     def _csg_from_evals(self, evals: np.ndarray) -> float:
#         # [n_class]
#         grads = evals[1:] - evals[:-1]
#         ratios = grads / (np.array(list(reversed(range(1, grads.shape[0] + 1)))) + 1)
#         csg: float = np.maximum.accumulate(ratios, -1).sum(0)
#         return csg

# # Datos de ejemplo
# if __name__ == "__main__":
#     from sklearn.datasets import load_iris
#     data = load_iris()
#     X = data.data
#     y = data.target

#     # Inicializar y ajustar el estimador
#     estimator = CumulativeGradientEstimator(M_sample=5000, k_nearest=5, distance="euclidean")
#     estimator.fit(X, y)

#     # Imprimir resultados
#     log.info(f"Evals: {estimator.evals}")
#     log.info(f"Evecs: {estimator.evecs}")
#     log.info(f"CSG: {estimator.csg}")

# import matplotlib.pyplot as plt

# def plot_eigenvalues(evals):
#     plt.figure(figsize=(10, 6))
#     plt.plot(evals, 'bo-', label="Eigenvalues")
#     plt.title("Eigenvalues of the Laplacian")
#     plt.xlabel("Index")
#     plt.ylabel("Eigenvalue")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Llama a la función de visualización
# plot_eigenvalues(estimator.evals)


# import pickle

# with open('estimator.pkl', 'wb') as f:
#     pickle.dump(estimator, f)

# print("El objeto estimator ha sido guardado en 'estimator.pkl'.")
