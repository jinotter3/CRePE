#!/usr/bin/env python
"""Train the CRePE adapter.

    python scripts/train.py fit --config configs/train_crepe.yaml

All options can be overridden on the command line, e.g.
``--data.batch_size 4 --trainer.devices 2``.
"""

import sys
from pathlib import Path

import torch
from lightning.pytorch.cli import LightningCLI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crepe.train.data import PanShotDataModule  # noqa: E402
from crepe.train.module import CRePETrainingModule  # noqa: E402


def main():
    # Same global numerics as the runs that produced the released checkpoints.
    torch.set_float32_matmul_precision("high")
    LightningCLI(CRePETrainingModule, PanShotDataModule, save_config_kwargs={"overwrite": True},
                 parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    main()
