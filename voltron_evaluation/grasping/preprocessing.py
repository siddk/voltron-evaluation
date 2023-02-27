"""
preprocessing.py

Utilities for loading and compiling the ARC Grasp Affordance Prediction Dataset into a Lightning DataModule.
"""
import logging
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torchvision.transforms.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Grab Logger
overwatch = logging.getLogger(__file__)


class GraspingRGBDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        examples: List[str],
        transforms: Callable[[torch.Tensor], torch.Tensor],
        pad_resolution: Tuple[int, int] = (640, 640),
        input_resolution: Tuple[int, int] = (224, 224),
        label_scale_factor: float = 0.125,
    ) -> None:
        super().__init__()
        self.data_dir, self.examples, self.label_scale_factor = data_dir, examples, label_scale_factor
        self.pad_resolution, self.input_resolution = pad_resolution, input_resolution
        self.transforms = transforms

        # Iterate through `self.examples` and retrieve input RGB & labels
        self.input_rgbs, self.labels = [], []
        for _idx, example in tqdm(enumerate(self.examples), total=len(self.examples), leave=False):
            # Read Images
            rgb_raw = read_image(str(self.data_dir / "color-input" / f"{example}.png"))
            lbl_raw = read_image(str(self.data_dir / "label" / f"{example}.png"))

            # Pad to Square --> separate handling for RGB and Labels:
            #   -> RGB :: Pad with Black (RGB 0-0-0) to `self.pad_resolution`
            #   -> Label :: Note `label` has 128 for POSITIVE, 0 for NEGATIVE, 255 for UNLABELED, so pad with 255!
            rgb_pad = F.pad(
                rgb_raw,
                [(self.pad_resolution[-1] - rgb_raw.size(-1)) // 2, (self.pad_resolution[-2] - rgb_raw.size(-2)) // 2],
                fill=0,
                padding_mode="constant",
            )
            lbl_pad = F.pad(
                lbl_raw,
                [(self.pad_resolution[-1] - lbl_raw.size(-1)) // 2, (self.pad_resolution[-2] - lbl_raw.size(-2)) // 2],
                fill=255,
                padding_mode="constant",
            )

            # Downsample & Bucketize `labels` --> 0 = NEGATIVE, 1 = POSITIVE, 2 = IGNORE
            lbl_classes = torch.round((lbl_pad.float() / 255) * 2.0).byte()
            lbl = F.resize(
                lbl_classes,
                [int(r * self.label_scale_factor) for r in self.pad_resolution],
                interpolation=InterpolationMode.NEAREST,
            ).squeeze()
            uniq = lbl.unique()

            # Label Assertions --> Labels can be in {0, 1, 2}, {0, 2}, {1, 2}, or {2}; if the latter, skip example!
            if torch.equal(uniq, torch.tensor([2]).byte()):
                continue

            # Otherwise, continue with assertions...
            assert lbl.size() == torch.Size([80, 80]) and lbl.dtype == torch.uint8
            assert (
                torch.equal(uniq, torch.tensor([0, 1, 2]).byte())
                or torch.equal(uniq, torch.tensor([0, 2]).byte())
                or torch.equal(uniq, torch.tensor([1, 2]).byte())
            )

            # Resize Padded RGB to `self.input_resolution`
            rgb = F.resize(rgb_pad, size=[r for r in self.input_resolution])
            assert rgb.size() == torch.Size([3, 224, 224]) and rgb.dtype == torch.uint8

            # Add to Trackers --> Note that we apply `transforms` *in the Dataset initializer*
            self.input_rgbs.append(self.transforms(rgb))
            self.labels.append(lbl.long())

        # Tensorize...
        self.input_rgbs, self.labels = torch.stack(self.input_rgbs), torch.stack(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_rgbs[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.input_rgbs)


class GraspingRGBDataModule(LightningDataModule):
    def __init__(
        self,
        data: Path,
        fold: int,
        n_val: int,
        bsz: int,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        pad_resolution: Tuple[int, int] = (640, 640),
        input_resolution: Tuple[int, int] = (224, 224),
        label_scale_factor: float = 0.125,
    ) -> None:
        super().__init__()
        self.data, self.fold, self.n_val, self.bsz, self.preprocess = data, fold, n_val, bsz, preprocess
        self.pad_resolution, self.input_resolution = pad_resolution, input_resolution
        self.label_scale_factor = label_scale_factor

        # Final "shapes" (for use in initializing segmentation models)
        self.label_shape = [int(r * self.label_scale_factor) for r in self.pad_resolution]

        # Read image IDs from `train-split.txt`
        with open(self.data / "train-split.txt", "r") as f:
            # Create train and validation splits by just sorting and splitting (based on fold)
            self.fit_examples = sorted([line.strip() for line in f.readlines()])
            self.train_examples = self.fit_examples[: (fold * self.n_val)] + self.fit_examples[(fold + 1) * self.n_val :]
            self.val_examples = self.fit_examples[(fold * self.n_val) : (fold + 1) * self.n_val]

        # Read image IDs from `test-split.txt`
        with open(self.data / "test-split.txt", "r") as f:
            self.test_examples = sorted([line.strip() for line in f.readlines()])

    def setup(self, stage: str) -> None:
        # Create Train & Validation Datasets
        if stage == "fit":
            overwatch.info("Compiling Grasp Train Dataset")
            self.train_dataset = GraspingRGBDataset(
                self.data,
                self.train_examples,
                self.preprocess,
                self.pad_resolution,
                self.input_resolution,
                self.label_scale_factor,
            )

            overwatch.info("Compiling Grasp Validation Dataset")
            self.val_dataset = GraspingRGBDataset(
                self.data,
                self.val_examples,
                self.preprocess,
                self.pad_resolution,
                self.input_resolution,
                self.label_scale_factor,
            )

        # Create Test Dataset
        if stage == "test":
            overwatch.info("Compiling Grasp Test Dataset")
            self.test_dataset = GraspingRGBDataset(
                self.data,
                self.test_examples,
                self.preprocess,
                self.pad_resolution,
                self.input_resolution,
                self.label_scale_factor,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.bsz, num_workers=8, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=512, num_workers=8, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=512, num_workers=8, shuffle=False)


def build_datamodule(
    data: Path, fold: int, n_val: int, bsz: int, preprocess: Callable[[torch.Tensor], torch.Tensor]
) -> LightningDataModule:
    return GraspingRGBDataModule(data / "suction-based" / "data", fold, n_val, bsz, preprocess)
