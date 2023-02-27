"""
adapter.py

ARC Grasping Segmentation Adapter definition -- as a LightningModule; takes as input both the backbone, as well as the
randomly initialized MAP extractor.

For now, the structure of this adapter is contingent on a MAP (variable-length) extraction method; can be overridden!
"""
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from torch.optim import AdamW, Optimizer
from torchvision.transforms.functional import gaussian_blur

# Book-Keeping
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PuPDecoder(nn.Module):
    def __init__(
        self, embed_dim: int, n_classes: int, stages: int, stage_features: Tuple[int, ...], kernel_size: int = 3
    ) -> None:
        super().__init__()
        assert len(stage_features) == stages, "Need to properly specify PuP Decoder Parameters!"

        # Create PuP Blocks
        all_features = [embed_dim, *list(stage_features)]
        self.pup_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(pre, post, kernel_size=kernel_size, padding=1),
                    nn.BatchNorm2d(post),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                )
                for pre, post in zip(all_features, all_features[1:])
            ]
        )

        # Final output projection...
        self.proj = nn.Conv2d(all_features[-1], n_classes, kernel_size=kernel_size, padding=1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Feed patches through PuP Decoder, getting final features [out_resolution, out_resolution, n_classes]."""
        for p in self.pup_blocks:
            grid = p(grid)

        # Project final grid to n_classes with a Conv2D
        return self.proj(grid)


class SegmenterPuP(LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        extractor: nn.Module,
        input_resolution: int = 224,
        output_resolution: int = 80,
        n_classes: int = 3,
        pup_stages: int = 4,
        pup_features: Tuple[int, ...] = (128, 64, 32, 16),
        unlabeled_index: int = 2,
    ) -> None:
        super().__init__()
        self.input_resolution, self.output_resolution, self.n_classes = input_resolution, output_resolution, n_classes
        self.pup_stages, self.pup_features = pup_stages, pup_features
        self.grid_size = int(extractor.n_latents**0.5)

        # Set weights per class label, ignoring the "unlabeled" regions (ignore_index)
        masked_label_weights = torch.ones(self.n_classes)
        masked_label_weights[unlabeled_index] = 0
        self.register_buffer("label_weights", masked_label_weights)

        # Create Network
        self.backbone, self.extractor = backbone, extractor
        self.pup_decoder = PuPDecoder(
            extractor.embed_dim, self.n_classes, stages=self.pup_stages, stage_features=self.pup_features
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Run through Backbone --> [bsz, n_patches, embed_dim]
        with torch.no_grad():
            patches = self.backbone(img, language=["picking something up"])

        # Extract Features --> reshape to Grid for PuPDecoder
        extracted = self.extractor(patches)
        grid = rearrange(extracted, "bsz (h w) d -> bsz d h w", h=self.grid_size, w=self.grid_size)

        # Get Segmentation logits
        return self.pup_decoder(grid)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Unpack batch of (image frame, per-pixel labels) --> run through model, and compute spatial cross-entropy."""
        img, labels = batch

        # Run forward pass --> [bsz, 3, 80, 80]
        logits = self.forward(img)

        # Compute Spatial Cross-Entropy subject to self.label_weights (weight of 0 corresponds to "unlabeled" regions)
        loss = F.cross_entropy(logits, labels, weight=self.label_weights)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Unpack batch of RGB frame, per-pixel labels, run through model, compute various ARC metrics."""
        img, labels = batch

        # Run forward pass --> [bsz, 3, 80, 80]
        logits = self.forward(img)

        # Compute Spatial Cross-Entropy subject to self.label_weights (weight of 0 corresponds to "unlabeled" regions)
        loss = F.cross_entropy(logits, labels, weight=self.label_weights)

        # Compute additional ARC Grasping Metrics -> Top-1 PR, Top-1% PR, Top-5% PR, Top-10% PR
        #   => Get probabilities (softmax) from `logits` --> [bsz, n_classes, 80, 80]
        #   => Smooth probabilities w/ Gaussian Blur with square kernel of size 29, sigma = 7
        #      - https://github.com/andyzeng/arc-robot-vision/blob/master/suction-based-grasping/convnet/evaluate.m#L46
        #      - `imgaussfilt` -> PyTorch via Matlab doc: https://www.mathworks.com/help/images/ref/imgaussfilt.html
        #   => Extract probabilities of "POSITIVE" labels (idx = 1) --> [bsz, 80, 80]
        #   => Get Top-1 (max), or 99th, 95th, 90th percentiles...
        #   => Precision Calculations!
        NEG_IDX, POS_IDX = 0, 1
        probs = torch.softmax(logits, dim=2)
        smoothed_probs = gaussian_blur(probs, kernel_size=29, sigma=7)
        positive_probs = smoothed_probs[:, POS_IDX]

        # Reshape positive probabilities for predictions & labels for ease of comparison...
        predicted_probs = rearrange(positive_probs, "bsz h w -> bsz (h w)")
        target_labels = rearrange(labels, "bsz h w -> bsz (h w)")

        # Top-1, Percentile Threshold Calculations
        top1_threshold, _ = torch.max(predicted_probs - 0.0001, dim=1)
        p99th_threshold = torch.quantile(predicted_probs, q=0.99, dim=1)
        p95th_threshold = torch.quantile(predicted_probs, q=0.95, dim=1)
        p90th_threshold = torch.quantile(predicted_probs, q=0.90, dim=1)

        # Compute PR @ X
        metrics = {"val_loss": loss}
        for m, t in [
            ("top1", top1_threshold),
            ("p99", p99th_threshold),
            ("p95", p95th_threshold),
            ("p90", p90th_threshold),
        ]:
            true_pos = (torch.gt(predicted_probs, t[..., None]) & torch.eq(target_labels, POS_IDX)).float()
            false_pos = (torch.gt(predicted_probs, t[..., None]) & torch.eq(target_labels, NEG_IDX)).float()

            # Write Precision --> use 1e-8 as eps...
            metrics[m] = true_pos.sum() / (true_pos.sum() + false_pos.sum() + 1e-8)

        # Log...
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, labels = batch

        # Run forward pass --> [bsz, 3, 80, 80]
        logits = self.forward(img)

        # Compute Spatial Cross-Entropy subject to self.label_weights (weight of 0 corresponds to "unlabeled" regions)
        loss = F.cross_entropy(logits, labels, weight=self.label_weights)

        # Compute additional ARC Grasping Metrics -> Top-1 PR, Top-1% PR, Top-5% PR, Top-10% PR
        #   => Get probabilities (softmax) from `logits` --> [bsz, n_classes, 80, 80]
        #   => Smooth probabilities w/ Gaussian Blur with square kernel of size 29, sigma = 7
        #      - https://github.com/andyzeng/arc-robot-vision/blob/master/suction-based-grasping/convnet/evaluate.m#L46
        #      - `imgaussfilt` -> PyTorch via Matlab doc: https://www.mathworks.com/help/images/ref/imgaussfilt.html
        #   => Extract probabilities of "POSITIVE" labels (idx = 1) --> [bsz, 80, 80]
        #   => Get Top-1 (max), or 99th, 95th, 90th percentiles...
        #   => Precision Calculations!
        NEG_IDX, POS_IDX = 0, 1
        probs = torch.softmax(logits, dim=2)
        smoothed_probs = gaussian_blur(probs, kernel_size=29, sigma=7)
        positive_probs = smoothed_probs[:, POS_IDX]

        # Reshape positive probabilities for predictions & labels for ease of comparison...
        predicted_probs = rearrange(positive_probs, "bsz h w -> bsz (h w)")
        target_labels = rearrange(labels, "bsz h w -> bsz (h w)")

        # Top-1, Percentile Threshold Calculations
        top1_threshold, _ = torch.max(predicted_probs - 0.0001, dim=1)
        p99th_threshold = torch.quantile(predicted_probs, q=0.99, dim=1)
        p95th_threshold = torch.quantile(predicted_probs, q=0.95, dim=1)
        p90th_threshold = torch.quantile(predicted_probs, q=0.90, dim=1)

        # Compute PR @ X
        metrics = {"test_loss": loss}
        for m, t in [
            ("test_top1", top1_threshold),
            ("test_p99", p99th_threshold),
            ("test_p95", p95th_threshold),
            ("test_p90", p90th_threshold),
        ]:
            true_pos = (torch.gt(predicted_probs, t[..., None]) & torch.eq(target_labels, POS_IDX)).float()
            false_pos = (torch.gt(predicted_probs, t[..., None]) & torch.eq(target_labels, NEG_IDX)).float()

            # Write Precision --> use 1e-8 as eps...
            metrics[m] = true_pos.sum() / (true_pos.sum() + false_pos.sum() + 1e-8)

        # Log...
        self.log_dict(metrics)

    def configure_optimizers(self) -> Optimizer:
        return AdamW([p for p in self.parameters() if p.requires_grad])


def instantiate_segmenter(backbone: nn.Module, extractor: nn.Module) -> LightningModule:
    return SegmenterPuP(backbone, extractor)
