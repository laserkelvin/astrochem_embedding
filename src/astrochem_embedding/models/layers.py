"""
layers.py

This module is intended for storing the building blocks
of models.
"""

import torch
from torch import Tensor  # this is used for type annotations
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from astrochem_embedding import layers


class VarianceHinge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding):
        # variance throughout the batch for each embedding dimension
        variance = 1.0 - torch.sqrt(embedding.var(dim=0) + 1e-10)
        # average along embedding dimensions
        return torch.maximum(torch.zeros_like(variance), variance).mean()


class CovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding) -> float:
        # covariance along embedding dimensions
        covariance = embedding.T.cov()
        mask = ~torch.eye(*covariance.shape, device=covariance.device, dtype=bool)
        # mask out the diagonal elements, we just want to minimize covariance
        loss = (covariance[mask] ** 2.0).mean()
        return loss


class VICRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.variance = VarianceHinge()
        self.covariance = CovarianceLoss()
        self.invariance = nn.MSELoss()

    def forward(self, z_1: torch.Tensor, z_2: torch.Tensor):
        packed = [z_1, z_2]
        variance = sum([self.variance(z) for z in packed])
        covariance = sum([self.covariance(z) for z in packed])
        invariance = self.invariance(z_1, z_2)
        return variance, invariance, covariance
