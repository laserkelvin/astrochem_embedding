
import torch
import pytorch_lightning as pl

from astrochem_embedding.pipeline.data import SELFIESData
from astrochem_embedding import get_paths
from astrochem_embedding.models import models

pl.seed_everything(215015)

BATCH_SIZE = 64
NUM_WORKERS = 0
EMBEDDING_DIM = 128
Z_DIM = 32
NUM_LAYERS = 1
LR = 1e-4

model = models.GRUAutoEncoder(EMBEDDING_DIM, Z_DIM, NUM_LAYERS, lr=LR)

data = SELFIESData(BATCH_SIZE, NUM_WORKERS)

logger = pl.loggers.TensorBoardLogger("tb_logs", name="AstrochemEmbedder")
summarizer = pl.callbacks.ModelSummary(max_depth=-1)

trainer = pl.Trainer(max_epochs=5, callbacks=[summarizer], gpus=1, logger=logger, precision=32)

torch.cuda.empty_cache()

trainer.fit(model, data)

paths = get_paths()

trainer.save_checkpoint(paths.get("models").joinpath("GRUAutoEncoder.ckpt"))
