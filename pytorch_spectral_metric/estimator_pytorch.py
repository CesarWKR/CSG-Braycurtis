# import time
# import torch
# import numpy as np
# from scipy.sparse.csgraph import laplacian
# from numpy.linalg import LinAlgError
# from lib_pytorch import find_samples, compute_expectation_with_monte_carlo
# from itertools import product
# import logging
# from logging import log

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class CumulativeGradientEstimator(object):
#     def __init__(self, M_sample=250, k_nearest=10, distance="euclidean"):
#         self.M_sample = M_sample  # Tamaño máximo de muestra por clase
#         self.k_nearest = k_nearest
#         self.distance = distance
#         self.P = {}  # Initialize P attribute
#         self.M = None  # Initialize M attribute

#     def fit(self, data, target):
#         print("Inicio de fit...", flush=True)
#         start_time = time.time()
        
#         data = data.to(device)
#         target = target.to(device)
#         self.n_class = torch.max(target) - min(0, torch.min(target)) + 1
#         print(f"Number of classes (n_class): {self.n_class}", flush=True)

#         # Pasar self.M_sample a find_samples
#         print("Llamando a find_samples...", flush=True)
#         class_samples, self.class_indices = find_samples(
#             data, target, self.n_class, M=self.M_sample
#         )

#         print("Llamando a compute...", flush=True)
#         self.compute(data, target, class_samples)
        
#         end_time = time.time()
#         print(f"Tiempo total de fit: {end_time - start_time} segundos", flush=True)
#         return self

#     def compute(self, data, target, class_samples):
#         start_time = time.time()
#         self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
#             data,
#             target,
#             class_samples,
#             class_indices=self.class_indices,
#             n_class=self.n_class,
#             k_nearest=self.k_nearest,
#             distance=self.distance,
#         )
#         end_time = time.time()
#         print(f"Tiempo de compute_expectation_with_monte_carlo: {end_time - start_time} segundos", flush=True)

#         # Verificar claves de similarity_arrays
#         print(f"Claves en similarity_arrays: {self.similarity_arrays.keys()}", flush=True)

#         start_time = time.time()
#         self.W = torch.eye(self.n_class, device=device)
#         for i, j in product(range(self.n_class), range(self.n_class)):
#             self.W[i, j] = 1 - torch.cdist(self.S[i].unsqueeze(0), self.S[j].unsqueeze(0), p=2).item()
#         end_time = time.time()
#         print(f"Tiempo de cálculo de W: {end_time - start_time} segundos", flush=True)
#         print(f"Tiempo de cálculo de W: {(end_time - start_time) * 0.016667} minutos", flush=True)

#         start_time = time.time()
#         self.difference = 1 - self.W

#         self.L_mat, dd = laplacian(self.W.cpu().numpy(), False, True)
#         try:
#             self.evals, self.evecs = torch.linalg.eigh(torch.tensor(self.L_mat, device=device))
#             self.csg = self._csg_from_evals(self.evals)
#         except LinAlgError as e:
#             log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")
#             self.evals = torch.ones([self.n_class], device=device) * np.nan
#             self.evecs = torch.ones([self.n_class, self.n_class], device=device) * np.nan
#             self.csg = np.nan
#         end_time = time.time()
#         print(f"Tiempo de cálculo de eigenvalues y eigenvectors: {end_time - start_time} segundos", flush=True)
#         print(f"Tiempo de cálculo de eigenvalues y eigenvectors: {(end_time - start_time) * 0.016667} minutos", flush=True)

#         # Compute P matrix for class comparisons
#         start_time = time.time()
#         self.P = torch.zeros([self.n_class, self.n_class], device=device)
#         for i in range(self.n_class):
#             for j in range(self.n_class):
#                 if i in self.similarity_arrays and j in self.similarity_arrays[i]:
#                     similarity_array = self.similarity_arrays[i][j]
#                     print(f"Calculando P[{i}, {j}] con {similarity_array.sample_probability_norm}", flush=True)
#                     self.P[i, j] = torch.sum(similarity_array.sample_probability_norm)
#                 else:
#                     print(f"Missing key: similarity_arrays[{i}][{j}]", flush=True)
#         end_time = time.time()
#         print(f"Tiempo de cálculo de P matrix: {end_time - start_time} segundos", flush=True)
#         print(f"Tiempo de cálculo de P matrix: {(end_time - start_time) * 0.016667} minutos", flush=True)

#         # Compute M matrix for all sample comparisons
#         start_time = time.time()
#         num_samples = data.shape[0]
#         self.M = torch.zeros([num_samples, num_samples], device=device)
#         for i in range(num_samples):
#             for j in range(num_samples):
#                 self.M[i, j] = 1 - torch.cdist(data[i].unsqueeze(0), data[j].unsqueeze(0), p=2).item()
#         end_time = time.time()
#         print(f"Tiempo de cálculo de M matrix: {end_time - start_time} segundos", flush=True)
#         print(f"Tiempo de cálculo de M matrix: {(end_time - start_time) * 0.016667} minutos", flush=True)

#     def _csg_from_evals(self, evals):
#         n = len(evals)
#         csg = torch.zeros(n, device=device)
#         for k in range(1, n):
#             csg[k] = torch.sum(evals[1:k + 1]) / torch.sum(evals[1:])
#         return csg






    # def _csg_from_evals(self, evals):
    #     n = len(evals)
    #     return sum([1 / (evals[i] - evals[i + 1]) for i in range(n - 1)]) / (n - 1)


import time
import torch
import numpy as np
from scipy.spatial.distance import cdist, braycurtis
from scipy.sparse.csgraph import laplacian
from numpy.linalg import LinAlgError
from lib_pytorch import find_samples, compute_expectation_with_monte_carlo
from itertools import product
import logging
from logging import log

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CumulativeGradientEstimator(object):
    def __init__(self, M_sample=250, k_nearest=10, distance="braycurtis"):
        self.M_sample = M_sample  # Tamaño máximo de muestra por clase
        self.k_nearest = k_nearest
        self.distance = distance
        self.P = {}  # Initialize P attribute
        self.M = None  # Initialize M attribute

    def fit(self, data, target):
        print("Inicio de fit...", flush=True)
        start_time = time.time()
        
        data = data.to(device)
        target = target.to(device)
        self.n_class = torch.max(target) - min(0, torch.min(target)) + 1
        print(f"Number of classes (n_class): {self.n_class}", flush=True)

        # Pasar self.M_sample a find_samples
        print("Llamando a find_samples...", flush=True)
        class_samples, self.class_indices = find_samples(
            data, target, self.n_class, M=self.M_sample
        )

        print("Llamando a compute...", flush=True)
        self.compute(data, target, class_samples)
        
        end_time = time.time()
        print(f"Tiempo total de fit: {end_time - start_time} segundos", flush=True)
        return self

    def compute(self, data, target, class_samples):
        start_time = time.time()
        self.S, self.similarity_arrays = compute_expectation_with_monte_carlo(
            data,
            target,
            class_samples,
            class_indices=self.class_indices,
            n_class=self.n_class,
            k_nearest=self.k_nearest,
            distance=self.distance,
        )
        end_time = time.time()
        print(f"Tiempo de compute_expectation_with_monte_carlo: {end_time - start_time} segundos", flush=True)

        # Verificar claves de similarity_arrays
        #print(f"Claves en similarity_arrays: {self.similarity_arrays.keys()}", flush=True)

        start_time = time.time()
        self.W = torch.eye(self.n_class, device=device)
        for i, j in product(range(self.n_class), range(self.n_class)):
            if self.distance == "braycurtis":
                self.W[i, j] = 1 - braycurtis(self.S[i].cpu().numpy(), self.S[j].cpu().numpy())
            else:
                self.W[i, j] = 1 - torch.cdist(self.S[i].unsqueeze(0), self.S[j].unsqueeze(0), p=2).item()
        end_time = time.time()
        print(f"Tiempo de cálculo de W: {end_time - start_time} segundos", flush=True)
        print(f"Tiempo de cálculo de W: {(end_time - start_time) * 0.016667} minutos", flush=True)

        start_time = time.time()
        self.difference = 1 - self.W

        self.L_mat, dd = laplacian(self.W.cpu().numpy(), False, True)
        try:
            self.evals, self.evecs = torch.linalg.eigh(torch.tensor(self.L_mat, device=device))
            self.csg = self._csg_from_evals(self.evals)
        except LinAlgError as e:
            log.warning(f"{str(e)}; assigning `evals,evecs,csg` to NaN")
            self.evals = torch.ones([self.n_class], device=device) * np.nan
            self.evecs = torch.ones([self.n_class, self.n_class], device=device) * np.nan
            self.csg = np.nan
        end_time = time.time()
        print(f"Tiempo de cálculo de eigenvalues y eigenvectors: {end_time - start_time} segundos", flush=True)
        print(f"Tiempo de cálculo de eigenvalues y eigenvectors: {(end_time - start_time) * 0.016667} minutos", flush=True)

        # Compute P matrix for class comparisons
        start_time = time.time()
        self.P = torch.zeros([self.n_class, self.n_class], device=device)
        for i in range(self.n_class):
            for j in range(self.n_class):
                if i in self.similarity_arrays and j in self.similarity_arrays[i]:
                    similarity_array = self.similarity_arrays[i][j]
                    self.P[i, j] = torch.sum(similarity_array.sample_probability_norm)
        end_time = time.time()
        print(f"Tiempo de cálculo de P matrix: {end_time - start_time} segundos", flush=True)
        print(f"Tiempo de cálculo de P matrix: {(end_time - start_time) * 0.016667} minutos", flush=True)

        # Compute M matrix for all sample comparisons
        start_time = time.time()
        num_samples = data.shape[0]
        self.M = torch.zeros([num_samples, num_samples], device=device)
        if self.distance == "braycurtis":
            data_cpu = data.cpu().numpy()  # Transfer data to CPU for Bray-Curtis distance calculation
            self.M = torch.tensor(1 - cdist(data_cpu, data_cpu, metric='braycurtis'), device=device)
        else:
            for i in range(num_samples):
                self.M[i, :] = 1 - torch.cdist(data[i].unsqueeze(0), data, p=2)
        end_time = time.time()
        print(f"Tiempo de cálculo de M matrix: {end_time - start_time} segundos", flush=True)
        print(f"Tiempo de cálculo de M matrix: {(end_time - start_time) * 0.016667} minutos", flush=True)

    def _csg_from_evals(self, evals):
        n = len(evals)
        csg = torch.zeros(n, device=device)
        for k in range(1, n):
            csg[k] = torch.sum(evals[1:k + 1]) / torch.sum(evals[1:])
        return csg

