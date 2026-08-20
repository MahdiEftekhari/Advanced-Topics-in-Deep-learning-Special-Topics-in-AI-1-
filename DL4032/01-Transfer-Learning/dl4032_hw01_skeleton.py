#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DL4032 - HW01 : Transfer Learning Optimization
Normalization Techniques and Gradient Dynamics
==============================================================================
STUDENT SKELETON

Fill in every `TODO`. The signatures below match the handout exactly -- do not
change them, since the autograder / TA scripts call them by name. You are free
to add helper functions.

Suggested order of work (one section per step of the handout):

    Step 1  Data preparation            sections 1.1 - 1.4
    Step 2  Base model setup            sections 2.1 - 2.4
    Step 3  Normalization variants      sections 3.1 - 3.4
    Step 4  Gradient infrastructure     sections 4.1 - 4.4
    Step 5  Training framework          sections 5.1 - 5.4
    Step 6  Loss landscapes             sections 6.1 - 6.4
    Step 7  Run the six experiments     section  7
    Step 8  Analysis and visualization  section  8
    Step 9  Report                      section  9 (write the PDF separately)

Checkpoints to hit before moving on:
    * after Step 2: a random batch produces a (N, 10) output and gradients reach
      the head but NOT the frozen trunk parameters;
    * after Step 3: all three heads run on the same batch without shape errors;
    * after Step 4: gradient norms are non-zero and change between steps;
    * after Step 5: one epoch on a 512-image subset completes in a few minutes.

Tip: run with `--smoke` constantly. Do not launch the full 6 x 15-epoch sweep
until the smoke run is clean end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# 0. Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch so the six runs are comparable.

    TODO: seed random, np.random, torch, torch.cuda; set cudnn.deterministic.
    Why it matters: the only thing that should differ between the six
    experiments is the normalizer and the clipping flag.
    """
    raise NotImplementedError


def get_device(prefer: str = "cuda") -> torch.device:
    """TODO: return cuda if available, else cpu."""
    raise NotImplementedError


# =============================================================================
# 1. Data preparation                                              (Step 1)
# =============================================================================

@dataclass
class DataConfig:
    data_root: str = "./data"
    batch_size: int = 64
    img_size: int = 224
    val_fraction: float = 0.1
    num_workers: int = 2
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    test_subset: Optional[int] = None
    seed: int = 42


def compute_dataset_stats(data_root: str = "./data"):
    """Step 1.2 -- per-channel mean and std of the CIFAR-10 training split.

    TODO:
      1. load datasets.CIFAR10(train=True, download=True) WITHOUT a transform;
      2. convert `.data` (uint8, shape (N, 32, 32, 3)) to float in [0, 1];
      3. return (mean, std) as 3-tuples, averaged over N, H, W.
    """
    raise NotImplementedError


def build_transforms(cfg: DataConfig, normalize_to: str = "imagenet"):
    """Step 1.3 -- augmentation + normalization pipelines.

    TODO:
      * train: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor,
        Normalize;
      * eval: ToTensor, Normalize only (never augment the validation set).
    Think about, and justify in your report: should you normalize with the
    CIFAR-10 statistics you just computed, or with the ImageNet statistics the
    pre-trained trunk was fitted to?
    """
    raise NotImplementedError


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Step 1.3 -- train / validation / test loaders.

    TODO:
      1. build two views of the 50k training split: one with train transforms,
         one with eval transforms;
      2. split indices deterministically (torch.randperm with a seeded
         generator) into train / val according to cfg.val_fraction;
      3. wrap in Subset, honour cfg.*_subset for quick debugging runs;
      4. return the three DataLoaders (shuffle only the training one).
    """
    raise NotImplementedError


def visualize_samples_per_class(dataset, out_path: str, n_per_class: int = 5) -> None:
    """Step 1.4 -- grid of sample images, one row per class.

    TODO: remember to un-normalize before imshow, or everything will look grey.
    """
    raise NotImplementedError


def plot_class_distribution(dataset, out_path: str) -> Dict[str, int]:
    """Step 1.2 -- bar chart of images per class; return the counts."""
    raise NotImplementedError


# =============================================================================
# 2. Base model setup                                              (Step 2)
# =============================================================================

def load_base_model(pretrained: bool = True) -> nn.Module:
    """Step 2.1 / 2.2 -- pre-trained MobileNetV2 with the classifier removed.

    TODO: load mobilenet_v2 with ImageNet weights and return its `.features`
    module (output: 1280 channels). Print the architecture once and note in your
    report how many inverted-residual blocks it has.
    """
    raise NotImplementedError


def freeze_base_model(base: nn.Module, unfreeze_last_block: bool = True,
                      n_trainable_blocks: int = 2) -> None:
    """Step 2.2 -- freeze the trunk except the last convolutional block.

    TODO:
      1. set requires_grad = False on every trunk parameter;
      2. if unfreeze_last_block, re-enable the last `n_trainable_blocks`
         children of `features`.
    """
    raise NotImplementedError


def set_frozen_bn_eval(module: nn.Module) -> None:
    """Keep BatchNorm layers with frozen parameters in eval() mode.

    TODO: iterate over modules; if it is a BatchNorm and none of its own
    parameters require grad, call .eval() on it.
    Why: requires_grad=False does NOT stop a BatchNorm in train mode from
    updating running_mean / running_var. Skipping this silently degrades the
    frozen ImageNet features and is the most common bug in this assignment.
    """
    raise NotImplementedError


# =============================================================================
# 3. Normalization variants                                        (Step 3)
# =============================================================================

class BatchNormHead(nn.Module):
    """Head A -- pool -> FC(256) -> BatchNorm1d -> ReLU -> dropout -> FC(10)."""

    def __init__(self, input_features: int, dropout_rate: float = 0.5):
        super(BatchNormHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pool -> flatten -> fc1 -> bn1 -> relu -> dropout -> fc2
        raise NotImplementedError


class LayerNormHead(nn.Module):
    """Head B -- same topology, nn.LayerNorm instead of nn.BatchNorm1d.

    TODO: define global_pool, fc1, ln1 (LayerNorm over 256 features), dropout,
    fc2; implement forward. Keep the topology identical to Head A so the
    comparison is controlled.
    """

    def __init__(self, input_features: int, dropout_rate: float = 0.5):
        super(LayerNormHead, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FilterResponseNorm(nn.Module):
    """Filter Response Normalization (Singh & Krishnan, 2020) -- from scratch.

    Do NOT use any built-in normalization layer here. Implement:

        nu2   = mean over the spatial dims (H, W) of x**2      # no mean subtraction!
        x_hat = x * rsqrt(nu2 + |eps|)
        y     = gamma * x_hat + beta

    TODO:
      * learnable gamma and beta, shape (1, C, 1, 1);
      * eps as a buffer (or a learnable parameter -- the paper recommends this
        for small spatial sizes; say which you chose and why);
      * raise a clear error if the input is not 4-D.

    Question to answer in the report: FRN does not subtract the mean, so the
    output is not centred. What does that imply for a plain ReLU placed after
    it, and what does the paper propose instead?
    """

    def __init__(self, num_features: int, epsilon: float = 1e-6):
        super(FilterResponseNorm, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TLU(nn.Module):
    """Thresholded Linear Unit: max(x, tau) with a learnable per-channel tau.

    TODO: one parameter of shape (1, C, 1, 1), initialised to zero.
    """

    def __init__(self, num_features: int):
        super(TLU, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FRNHead(nn.Module):
    """Head C -- FRN (+ TLU) adaptation head.

    DESIGN DECISION you must make and justify: where do you put the FRN?
    FRN normalizes over (H, W). If you apply it after global average pooling,
    the spatial size is 1x1 and x / sqrt(mean(x^2)) reduces to sign(x) -- all
    magnitude information is destroyed. Think about which tensor in this head
    actually has spatial extent, and place the layer accordingly.

    TODO: build the head, keeping fc1(->256) / dropout / fc2(->10) so that the
    comparison against heads A and B stays controlled.
    """

    def __init__(self, input_features: int, dropout_rate: float = 0.5):
        super(FRNHead, self).__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def create_adaptation_head(norm_type: str, input_features: int,
                           dropout_rate: float = 0.5) -> nn.Module:
    """Step 3.2 -- factory. Provided; do not modify."""
    if norm_type == "batch":
        return BatchNormHead(input_features, dropout_rate)
    elif norm_type == "layer":
        return LayerNormHead(input_features, dropout_rate)
    elif norm_type == "frn":
        return FRNHead(input_features, dropout_rate)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


class TransferModel(nn.Module):
    """Step 3.3 -- frozen trunk + adaptation head.

    TODO in forward():
      1. resize the 32x32 input to self.input_size (F.interpolate on the GPU is
         much cheaper than resizing in the dataloader -- 224/32 = 7x per side,
         i.e. 49x the pixels moved per batch);
      2. run the trunk;
      3. run the head.
    Also override train() so that set_frozen_bn_eval() is re-applied every time
    the model is put back into training mode.
    """

    def __init__(self, base_model: nn.Module, adaptation_head: nn.Module,
                 input_size: int = 224):
        super(TransferModel, self).__init__()
        self.base_model = base_model
        self.adaptation_head = adaptation_head
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(norm_type: str, cfg: "ExperimentConfig", device) -> TransferModel:
    """Assemble trunk + head, freeze, move to device. Re-seed first."""
    raise NotImplementedError


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    raise NotImplementedError


def sanity_check_pipeline(device, batch_size: int = 4, img_size: int = 224) -> None:
    """Steps 2.4 and 3.4 -- validate all three variants on a random batch.

    TODO assert that:
      * output shape == (batch_size, 10);
      * loss.backward() populates grads on the head and on the unfrozen block;
      * NO frozen parameter received a gradient.
    """
    raise NotImplementedError


# =============================================================================
# 4. Gradient analysis infrastructure                              (Step 4)
# =============================================================================

class GradientTracker:
    """Step 4.1 -- record gradient statistics with hooks.

    Use Parameter.register_hook so you capture the raw gradient BEFORE the
    optimizer and BEFORE clipping. Storing every gradient tensor will exhaust
    RAM, so keep scalar norms every step and only subsample full values
    occasionally (see histogram_every / max_hist_values).
    """

    def __init__(self, model: nn.Module, tracked_layers: Optional[Sequence[str]] = None,
                 histogram_every: int = 50, max_hist_values: int = 20000):
        self.model = model
        self.gradients: Dict[str, List[float]] = {}
        self.grad_values: Dict[str, List[np.ndarray]] = {}
        self.global_norms: List[float] = []
        self.handles = []
        self.histogram_every = histogram_every
        self.max_hist_values = max_hist_values
        self.setup_hooks(tracked_layers)

    def setup_hooks(self, tracked_layers) -> None:
        """TODO: register a hook on every trainable parameter (filtered by
        tracked_layers if given) and keep the handles so they can be removed."""
        raise NotImplementedError

    def step(self) -> None:
        """TODO: call once per optimizer step; append per-layer norms and the
        global norm (sqrt of the sum of squared per-layer norms)."""
        raise NotImplementedError

    def reset_gradients(self) -> None:
        """TODO: clear all buffers."""
        raise NotImplementedError

    def remove_hooks(self) -> None:
        """TODO: remove every registered handle (leaking hooks leaks memory)."""
        raise NotImplementedError

    def layer_norm_summary(self) -> Dict[str, Dict[str, float]]:
        """TODO: per-layer mean / std / min / max / final gradient norm."""
        raise NotImplementedError

    def flat_values(self, name_filter: str = "") -> np.ndarray:
        """TODO: concatenate the sampled gradient values for the histograms."""
        raise NotImplementedError


def plot_gradient_histogram(trackers: Dict[str, GradientTracker], out_path: str,
                            name_filter: str = "adaptation_head") -> None:
    """Step 4.2a -- overlaid histograms of |gradient|.

    TODO: gradients span several orders of magnitude, so plot log10|g| (or use
    a log-scaled x axis); a linear histogram will be one spike at zero.
    """
    raise NotImplementedError


def plot_layerwise_gradient_norms(trackers: Dict[str, GradientTracker],
                                  out_path: str) -> None:
    """Step 4.2b -- mean gradient norm per layer, ordered trunk -> head."""
    raise NotImplementedError


def plot_gradient_evolution(trackers: Dict[str, GradientTracker], out_path: str,
                            smooth: int = 20) -> None:
    """Step 4.2c -- global gradient norm vs optimizer step (smoothed)."""
    raise NotImplementedError


def train_with_gradient_clipping(model, train_loader, optimizer, criterion,
                                 clip_value: float = 1.0, device: str = "cuda",
                                 gradient_tracker: Optional[GradientTracker] = None):
    """Step 4.3 -- one training epoch, optionally clipping the gradient norm.

    TODO:
      1. standard loop: zero_grad -> forward -> loss -> backward;
      2. if clip_value is not None, call torch.nn.utils.clip_grad_norm_ and keep
         its return value -- it is the PRE-clipping total norm, which is what you
         want to log;
      3. optimizer.step(), then gradient_tracker.step();
      4. return (avg_loss, accuracy, mean_pre_clip_norm, clip_rate) where
         clip_rate = fraction of steps whose norm exceeded the threshold.
         If the clip rate is ~0, clipping did nothing -- an important negative
         result for your report, not a bug.
    """
    raise NotImplementedError


# =============================================================================
# 5. Training framework                                            (Step 5)
# =============================================================================

@dataclass
class ExperimentConfig:
    name: str
    norm_type: str = "batch"
    use_grad_clip: bool = False
    clip_value: float = 1.0
    num_epochs: int = 15
    head_lr: float = 1e-3
    base_lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.5
    label_smoothing: float = 0.0
    img_size: int = 224
    unfreeze_last_block: bool = True
    seed: int = 42
    scheduler: str = "cosine"


def build_optimizer(model: TransferModel, cfg: ExperimentConfig):
    """TODO: parameter groups.

      * head parameters at cfg.head_lr;
      * unfrozen trunk parameters at cfg.base_lr (smaller -- you do not want to
        destroy pre-trained features);
      * exclude 1-D parameters (biases, gamma, beta, tau) from weight decay.
        Decaying a normalizer's scale towards zero would penalise the three
        heads unequally, since they have different numbers of such parameters.
    """
    raise NotImplementedError


@torch.no_grad()
def validate_model(model, val_loader, criterion, device: str = "cuda"):
    """Step 5.2 -- return (loss, accuracy, per_class_accuracy).

    TODO: remember model.eval(). For the BatchNorm head this switches to running
    statistics; comment in your report on whether that changes the train/val gap
    relative to LayerNorm and FRN, which behave identically in both modes.
    """
    raise NotImplementedError


def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs: int = 15, device: str = "cuda",
                use_grad_clip: bool = False, clip_value: float = 1.0,
                gradient_tracker: Optional[GradientTracker] = None,
                scheduler=None, checkpoint_path: Optional[str] = None,
                verbose: bool = True) -> Dict:
    """Step 5.1 -- full loop with logging.

    TODO: per epoch, record train_loss, train_acc, val_loss, val_acc,
    grad_norm, clip_rate, lr, epoch_time; track the best validation accuracy,
    save a checkpoint, and restore the best weights at the end (the loss
    landscape must be computed at a meaningful minimum, not at whatever the last
    epoch happened to produce).
    """
    raise NotImplementedError


class ExperimentManager:
    """Step 5.3 -- organise multiple runs."""

    def __init__(self, save_dir: str = "./experiments"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        for sub in ("checkpoints", "logs", "figures", "landscapes"):
            os.makedirs(os.path.join(save_dir, sub), exist_ok=True)
        self.experiments: Dict[str, Dict] = {}
        self.trackers: Dict[str, GradientTracker] = {}

    def run_experiment(self, name, model, train_loader, val_loader, **kwargs) -> Dict:
        """TODO: build criterion/optimizer/scheduler/tracker, call train_model,
        store {config, history, gradient_summary}, save to disk, return it."""
        raise NotImplementedError

    def evaluate_on_test(self, name, model, test_loader, device="cuda") -> Dict:
        """TODO: final test-set evaluation of the best checkpoint."""
        raise NotImplementedError

    def save_results(self) -> None:
        """TODO: write one JSON per experiment + results_summary.csv.
        Colab disconnects; save after every run, not at the end."""
        raise NotImplementedError

    def load_results(self, path: Optional[str] = None) -> Dict[str, Dict]:
        """TODO: reload the JSON logs so analysis can run without retraining."""
        raise NotImplementedError

    def summary_table(self) -> pd.DataFrame:
        """TODO: one row per experiment. Suggested columns: norm, clip,
        best_val_acc, best_epoch, final_train_acc, generalisation gap,
        mean/std gradient norm, clip rate, epochs-to-90%-of-best, wall time."""
        raise NotImplementedError


# =============================================================================
# 6. Loss landscape visualization                                  (Step 6)
# =============================================================================

def get_random_directions(model: nn.Module, only_head: bool = True,
                          seed: Optional[int] = None):
    """Step 6.2 -- two FILTER-NORMALIZED random directions (Li et al., 2018).

    TODO: for each parameter, draw d ~ N(0, I) and rescale it so that each
    filter/row of d has the norm of the corresponding filter/row of the weight:

        d[i] <- d[i] * ||w[i]|| / ||d[i]||

    Do NOT skip the normalization. BatchNorm and FRN make the loss invariant to
    rescaling of the weights, so a network with larger weights would appear
    artificially flatter and the comparison between your three heads would be
    meaningless.
    """
    raise NotImplementedError


def compute_loss_landscape(model, dataloader, criterion, device,
                           alpha_range=(-1, 1), beta_range=(-1, 1),
                           steps: int = 10, direction1=None, direction2=None,
                           only_head: bool = True, max_batches: int = 8) -> Dict:
    """Step 6.1 -- evaluate loss on the grid theta* + alpha*d1 + beta*d2.

    TODO:
      1. snapshot theta* (clone!) before perturbing anything;
      2. cache a FIXED set of batches so every grid point sees the same data --
         otherwise you are plotting sampling noise;
      3. for each (alpha, beta), write the perturbed weights in-place under
         torch.no_grad(), evaluate, store the loss;
      4. restore theta* at the end -- forgetting this silently corrupts the
         model for every subsequent step;
      5. return alphas, betas, losses (and, optionally, a scalar smoothness
         measure such as the mean |discrete Laplacian|).
    """
    raise NotImplementedError


def compute_head_loss_landscape(model, dataloader, criterion, device,
                                steps: int = 10, **kwargs) -> Dict:
    """Step 6.4 -- restrict the landscape to the adaptation-head parameters."""
    raise NotImplementedError


def plot_loss_landscape(landscape_data: Dict, title: str = "Loss Landscape",
                        out_path: Optional[str] = None) -> None:
    """Step 6.3 -- contour and/or 3-D surface, with the minimum marked.

    TODO: clip extreme values (e.g. at the 99th percentile) before plotting, or
    a single diverging corner will flatten the whole colour scale.
    """
    raise NotImplementedError


def plot_landscape_grid(landscapes: Dict[str, Dict], out_path: str) -> None:
    """Step 8.3 -- side-by-side landscapes on a SHARED colour scale.

    TODO: per-panel autoscaling makes every landscape look the same; use common
    contour levels across panels.
    """
    raise NotImplementedError


# =============================================================================
# 8. Comparative analysis                                          (Step 8)
# =============================================================================

def plot_training_curves(experiments: Dict[str, Dict], out_path: str) -> None:
    """Step 8.1 -- 2x2 grid: train/val loss and train/val accuracy.

    TODO: use colour for the normalizer and line style for clip on/off, so the
    six curves stay readable.
    """
    raise NotImplementedError


def plot_grad_norm_curves(experiments: Dict[str, Dict], out_path: str) -> None:
    """Step 8.2 -- per-epoch mean gradient norm and clip rate."""
    raise NotImplementedError


def plot_summary_bars(df: pd.DataFrame, out_path: str) -> None:
    """Step 8 -- grouped bars: accuracy, generalisation gap, gradient norm."""
    raise NotImplementedError


# =============================================================================
# 7 + 9. Orchestration                                       (Steps 7 and 9)
# =============================================================================

EXPERIMENT_CONFIGS = [
    {"name": "batchnorm_no_clip", "norm_type": "batch", "use_grad_clip": False},
    {"name": "batchnorm_clip", "norm_type": "batch", "use_grad_clip": True},
    {"name": "layernorm_no_clip", "norm_type": "layer", "use_grad_clip": False},
    {"name": "layernorm_clip", "norm_type": "layer", "use_grad_clip": True},
    {"name": "frn_no_clip", "norm_type": "frn", "use_grad_clip": False},
    {"name": "frn_clip", "norm_type": "frn", "use_grad_clip": True},
]


def run_all(args) -> None:
    """Step 7 -- the whole pipeline.

    TODO, in order:
      1. set_seed, get_device, build dataloaders;
      2. Step 1 artefacts: dataset statistics, sample grid, class distribution;
      3. sanity_check_pipeline();
      4. for each config in EXPERIMENT_CONFIGS: build model, run experiment,
         evaluate on test, compute the head loss landscape, free GPU memory;
      5. Step 8 comparative plots + summary table;
      6. write the executive summary skeleton for Step 9.
    """
    raise NotImplementedError


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DL4032 HW01")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--save-dir", default="./experiments")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--clip-value", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--val-subset", type=int, default=None)
    p.add_argument("--test-subset", type=int, default=None)
    p.add_argument("--landscape-steps", type=int, default=15)
    p.add_argument("--landscape-batches", type=int, default=6)
    p.add_argument("--only", default=None, help="comma-separated experiment names")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--skip-landscape", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.smoke:
        args.epochs = 1
        args.img_size = 96
        args.batch_size = 32
        args.train_subset = 512
        args.val_subset = 256
        args.test_subset = 256
        args.landscape_steps = 5
        args.landscape_batches = 2
        args.only = args.only or "batchnorm_no_clip,frn_clip"
        args.save_dir = os.path.join(args.save_dir, "_smoke")
    run_all(args)


if __name__ == "__main__":
    main()