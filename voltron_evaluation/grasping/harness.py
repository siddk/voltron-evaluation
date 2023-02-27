"""
harness.py

Class defining the evaluation harness for the ARC Grasp Affordance Prediction task; a GraspAffordanceHarness is
comprised of the following three parts:
    1) __init__  :: Takes backbone, factory function for extractor, factory function for adapter (as LightningModule)
    2) fit       :: Invokes train/fit protocol, with additional steps for k-fold cross validation, seeds, etc.
                    Uses a Trainer on top of the defined LightningModule --> simple calls to Trainer.fit().
    3) test      :: Function defining the testing (or test metric aggregation) procedure.

By default, assumes a MAP-based feature extractor --> PuP-style segmenter head; this can be overridden as you see fit!
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from voltron_evaluation.grasping.adapter import instantiate_segmenter
from voltron_evaluation.grasping.preprocessing import build_datamodule
from voltron_evaluation.util import LOG_CONFIG, set_global_seed

# Grab Logger
logging.config.dictConfig(LOG_CONFIG)
overwatch = logging.getLogger(__file__)


class GraspAffordanceHarness:
    def __init__(
        self,
        model_id: str,
        backbone: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        extractor_init_fn: Callable[[], nn.Module],
        segmenter_init_fn: Callable[[nn.Module, nn.Module], LightningModule] = instantiate_segmenter,
        run_dir: Path = Path("runs/evaluation/grasping"),
        data: Path = Path("data/grasping"),
        n_val: int = 170,
        n_folds: int = 5,
        bsz: int = 64,
        epochs: int = 50,
        seed: int = 7,
    ) -> None:
        overwatch.info("Initializing GraspAffordanceHarness")
        self.model_id, self.backbone, self.preprocess = model_id, backbone, preprocess
        self.extractor_init_fn, self.segmenter_init_fn = extractor_init_fn, segmenter_init_fn
        self.run_dir, self.data, self.n_val, self.n_folds = run_dir, data, n_val, n_folds
        self.bsz, self.epochs, self.seed = bsz, epochs, seed

        # Set Randomness
        set_global_seed(self.seed)

        # Create Run Directory
        os.makedirs(self.run_dir / self.model_id, exist_ok=True)

    def get_datamodule(self, fold: int = 0) -> LightningDataModule:
        return build_datamodule(self.data, fold, self.n_val, self.bsz, self.preprocess)

    def fit(self) -> None:
        overwatch.info("Invoking GraspAffordanceHarness.fit()")

        # Iterate through Folds
        for fold in range(self.n_folds):
            # Short-Circuit...
            if (self.run_dir / self.model_id / f"fold-{fold}+metrics.json").exists():
                continue

            # Run out the folds...
            overwatch.info(f"Starting Data Processing for Fold {fold + 1} / {self.n_folds}")
            dm = self.get_datamodule(fold)

            overwatch.info("Instantiating Adapter Model and Callbacks")
            segmenter = self.segmenter_init_fn(self.backbone, self.extractor_init_fn())
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(self.run_dir / self.model_id),
                filename=f"fold={fold}+" + "{epoch:02d}-{val_loss:0.4f}-{top1:0.4f}-{p99:0.4f}.pt",
                monitor="p99",
                mode="max",
                save_top_k=1,
            )

            overwatch.info("Training...")
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=self.epochs,
                log_every_n_steps=-1,
                logger=None,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(segmenter, datamodule=dm)

            # Get fold metrics & serialize...
            overwatch.info("Running final validation epoch with best model...")
            fold_metrics = trainer.validate(segmenter, datamodule=dm, ckpt_path="best")
            test_metrics = trainer.test(segmenter, datamodule=dm, ckpt_path="best")
            with open(self.run_dir / self.model_id / f"fold-{fold}+metrics.json", "w") as f:
                json.dump({"validation": fold_metrics[0], "test": test_metrics[0]}, f, indent=4)

    def test(self) -> None:
        overwatch.info("Compiling Grasp Affordance Test Metrics")
        best_val_loss, test_metrics = None, None
        for fold in range(self.n_folds):
            with open(self.run_dir / self.model_id / f"fold-{fold}+metrics.json", "r") as f:
                metrics = json.load(f)

            # Compare on Validation p99
            if best_val_loss is None or best_val_loss > metrics["validation"]["val_loss"]:
                best_val_loss, test_metrics = metrics["validation"]["val_loss"], metrics["test"]

        # Print Test Metrics!
        overwatch.info("Grasp Affordance Prediction =>> Test Metrics")
        for mname, mkey in [
            ("PR @ 90%", "test_p90"),
            ("PR @ 95%", "test_p95"),
            ("PR @ 99%", "test_p99"),
            ("PR @ Top-1", "test_top1"),
        ]:
            overwatch.info(f"\t{mname}: {test_metrics[mkey]:0.4f}")
