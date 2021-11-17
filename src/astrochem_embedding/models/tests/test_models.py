import torch

from astrochem_embedding.models import models


@torch.no_grad()
def test_gruautoencoder():
    model = models.GRUAutoEncoder(embedding_dim=8, z_dim=16, num_layers=1)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    output = model(test_array)


@torch.no_grad()
def test_gruautoencoder_multi():
    model = models.GRUAutoEncoder(embedding_dim=8, z_dim=16, num_layers=3)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    output = model(test_array)


@torch.no_grad()
def test_gruautoencoder_step():
    model = models.GRUAutoEncoder(embedding_dim=8, z_dim=16, num_layers=1)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    model.step((test_array, test_array, test_array), "test")


@torch.no_grad()
def test_vicgae():
    model = models.VICGAE(embedding_dim=8, z_dim=16, num_layers=1)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    output = model(test_array)


@torch.no_grad()
def test_vicgae_multi():
    model = models.VICGAE(embedding_dim=8, z_dim=16, num_layers=3)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    output = model(test_array)


@torch.no_grad()
def test_vicgae_step():
    model = models.VICGAE(embedding_dim=8, z_dim=16, num_layers=1)
    # expect a tensor of shape batch, seq_length
    test_array = torch.randint(0, 10, size=(32, 10))
    model.step((test_array, test_array, test_array), "test")


def test_load_gruautoencoder():
    model = models.GRUAutoEncoder.from_pretrained()


def test_load_vicgae():
    model = models.VICGAE.from_pretrained()
