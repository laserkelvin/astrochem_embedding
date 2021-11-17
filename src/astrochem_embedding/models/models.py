"""
models.py

This module is intended for *composed* models; i.e.
ready to training/usage, based off of layers defined in
either `torch`, other packages, or in `astrochem_embedding.layers`.
"""

from typing import Union, Iterable
from pathlib import Path

import torch
import joblib
from torch import Tensor  # this is used for type annotations
from torch import nn
from torch.nn import functional as F
import wandb
import pytorch_lightning as pl

from astrochem_embedding.models import layers
from astrochem_embedding import get_paths, get_pretrained_path, Translator


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-3,
        vocab_yaml: Union[str, None] = None,
    ):
        super().__init__()
        if not vocab_yaml:
            paths = get_paths()
            vocab_yaml = get_pretrained_path().joinpath("translator.yml")
        self.vocab = Translator.from_yaml(vocab_yaml)
        vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.metric = nn.BCELoss()
        self.example_input_array = torch.randint(0, vocab_size, size=(64, 10))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokens2smiles(self, tokens: Iterable[int]):
        return self.vocab.indices_to_smiles(tokens)

    @torch.no_grad()
    def embed_molecule(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X.unsqueeze_(0)
        # shape [N, S, D] for S sequence length and D features
        word_embeddings = self.embedding(X)
        mask = X != self.vocab.alphabet.index("[nop]")
        z = (
            torch.einsum("ijk,ij->ik", word_embeddings, mask.float())
            / self.hparams.embedding_dim
        )
        return z

    @torch.no_grad()
    def embed_smiles(self, smiles: str) -> torch.Tensor:
        labels, _ = self.vocab.tokenize_smiles(smiles)
        tokens = torch.LongTensor(labels)
        return self.embed_molecule(tokens.unsqueeze(0))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return opt

    def step(self, batch, prefix: str):
        X1, X2, Y = batch
        targets = F.one_hot(Y, num_classes=self.vocab_size)
        output = self(X1)
        loss = self.metric(output, targets.float())
        self.log("{prefix}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, "validation")
        # get some examples
        # ex_targets = labels[:10]
        # with torch.no_grad():
        #    ex_outputs = self(ex_targets)
        # ex_targets = ex_targets.cpu().numpy()
        # ex_outputs = ex_outputs.argmax(dim=-1).cpu().numpy()
        # target_smiles = [self.tokens2smiles(t) for t in ex_targets]
        # output_smiles = [self.tokens2smiles(t) for t in ex_outputs]
        # self.log("example_smiles",
        #    {"targets": target_smiles, "outputs": output_smiles}
        # )
        return loss

    @classmethod
    def from_pretrained(cls):
        pretrained_path = get_pretrained_path().joinpath(f"{cls.__name__}.pkl")
        return joblib.load(pretrained_path)


class GRUAutoEncoder(AutoEncoder):
    def __init__(
        self,
        embedding_dim: int,
        z_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        vocab_yaml: Union[str, None] = None,
    ):
        if not vocab_yaml:
            paths = get_paths()
            vocab_yaml = get_pretrained_path().joinpath("translator.yml")
        translator = Translator.from_yaml(vocab_yaml)
        vocab_size = len(translator)
        encoder = nn.GRU(
            embedding_dim,
            z_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        decoder = nn.GRU(
            z_dim, z_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        output = nn.Sequential(nn.Linear(z_dim, vocab_size), nn.Softmax(dim=-1))
        super().__init__(embedding_dim, encoder, decoder, lr, vocab_yaml)
        self.output = output
        self.save_hyperparameters()

    def forward(self, X):
        embeddings = self.embedding(X)
        z_o, z_h = self.encoder(embeddings)
        o_o, o_h = self.decoder(z_o, z_h)
        output = self.output(o_o)
        return output


class VICGAE(GRUAutoEncoder):
    def __init__(
        self,
        embedding_dim: int,
        z_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        vocab_yaml: Union[str, None] = None,
    ):
        super().__init__(embedding_dim, z_dim, num_layers, dropout, lr, vocab_yaml)
        self.vic_reg = layers.VICRegularization()

    def _vic_regularization(self, batch):
        X1, X2, Y = batch
        mask_1 = X1 != self.vocab.alphabet.index("[nop]")
        mask_2 = X2 != self.vocab.alphabet.index("[nop]")
        # shape [N, S, Z] for S sequence length
        word_z_1 = self.embedding(X1)
        word_z_2 = self.embedding(X2)
        # idea behind summmation is to have molecule = sum of characters
        z_1 = (
            torch.einsum("ijk,ij->ik", word_z_1, mask_1.float())
            / self.hparams.embedding_dim
        )
        z_2 = (
            torch.einsum("ijk,ij->ik", word_z_2, mask_2.float())
            / self.hparams.embedding_dim
        )
        v, i, c = self.vic_reg(z_1, z_2)
        return v, i, c

    def step(self, batch, prefix: str):
        X1, X2, Y = batch
        targets = F.one_hot(Y, num_classes=self.vocab_size).float()
        output_1 = self(X1)
        output_2 = self(X2)
        # include the VIC regularization
        v, i, c = self._vic_regularization(batch)
        loss_1 = self.metric(output_1, targets)
        loss_2 = self.metric(output_2, targets)
        loss = loss_1 + loss_2 + v + i + c
        self.log(f"{prefix}_loss", {"loss": loss, "v": v, "i": i, "c": c})
        return loss
