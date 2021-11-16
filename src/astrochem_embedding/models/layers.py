
"""
layers.py

This module is intended for storing the building blocks
of models.
"""

import torch
from torch import Tensor      # this is used for type annotations
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from astrochem_embedding import layers


class VarianceHinge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding):
        # variance throughout the batch for each embedding dimension
        variance = 1. - torch.sqrt(embedding.var(dim=0) + 1e-10)
        # average along embedding dimensions
        return torch.maximum(0., variance).mean()


class CovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding) -> float:
        # covariance along embedding dimensions
        covariance = embedding.T.cov()
        mask = ~torch.eye(*covariance.shape, device=covariance.device, dtype=bool)
        # mask out the diagonal elements, we just want to minimize covariance
        loss = covariance[mask]**2..mean()
        return loss
