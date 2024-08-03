from dataclasses import dataclass
import torch

Array = torch.Tensor

@dataclass
class SimilarityArrays:
    sample_probability: Array
    sample_probability_norm: Array

