from dataclasses import dataclass
import torch

Tensor = torch.Tensor

@dataclass
class SimilarityArrays:
    sample_probability: Tensor
    sample_probability_norm: Tensor

