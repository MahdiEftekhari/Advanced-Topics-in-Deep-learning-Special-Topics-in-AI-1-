# DL4032 HW01 — Reference Solution

**Learning material, not a submission.** This directory contains full working
implementations of every part the assignment asks for, plus the expected
results, so it doubles as an instructor/TA answer key and as reference material
for students studying the topic after the fact. If you are currently working on
this assignment, use `../dl4032_hw01_skeleton.py` and attempt it yourself first
— come back here to compare only once you've submitted your own solution.

---

## Contents

```
Solution/
├── dl4032_hw01_solution.py    complete reference implementation, Steps 1–9
└── README.md                  this file
```

`dl4032_hw01_solution.py` is a drop-in replacement for
`../dl4032_hw01_skeleton.py`: identical module layout, identical public
signatures, same CLI. Every skeleton section number maps 1:1 onto a solution
section number and onto a step of the handout, so a submission can be graded by
scrolling the two files side by side.

If you also keep a notebook version of the solution, put it here beside the
module and import from it (`from dl4032_hw01_solution import *`) rather than
copy-pasting the code — otherwise the two drift apart within a semester.

---

## Running it

```bash
pip install torch torchvision matplotlib numpy pandas

# ~2 min on a T4 — validates the entire pipeline end to end on 512 images
python dl4032_hw01_solution.py --smoke

# reduced preset, matches what students are told to use (~90 min on a T4)
python dl4032_hw01_solution.py --epochs 15 --img-size 160 \
    --train-subset 10000 --val-subset 2000 --save-dir ./run_reduced

# full assignment, 6 experiments × 15 epochs at 224×224 (~10–13 GPU-hours)
python dl4032_hw01_solution.py --epochs 15 --save-dir ./run_full

# re-run a single configuration
python dl4032_hw01_solution.py --only frn_clip --epochs 15
```

Useful flags: `--skip-eda`, `--skip-landscape`, `--keep-models`,
`--landscape-steps`, `--landscape-batches`, `--seed`.

Output tree under `--save-dir`:

```
checkpoints/<name>_best.pt      weights of the best-validation epoch
logs/<name>.json                per-epoch history + per-layer gradient summary
landscapes/<name>.npz           raw alpha/beta/loss grids
figures/01…09_*.png             every plot required by Step 8
results_summary.csv             one row per experiment
executive_summary.md            auto-filled scaffold for Step 9
dataset_stats.json              computed CIFAR-10 mean/std and class counts
```

`ExperimentManager.load_results()` reloads the JSON logs, so the whole Step 8
analysis can be regenerated without retraining.

---

## Verification status

The implementation was exercised end to end on synthetic data (CIFAR-10 and the
pretrained weights were not downloadable in the test sandbox). Checks that pass:

- FRN output has unit RMS over the spatial axes when γ=1, β=0;
- all three heads emit `(N, 10)` and receive gradients on every parameter;
- freezing leaves the trunk partially trainable and puts 48 frozen BN layers into
  `eval()`;
- no frozen parameter ever receives a gradient;
- the gradient tracker records exactly `epochs × steps_per_epoch` global norms;
- landscape directions are filter-normalised row by row;
- model parameters are restored **bit-exactly** after a landscape sweep;
- every figure renders and `save_results` / `load_results` round-trip.

Before the first run of the semester, do one `--smoke` on real CIFAR-10 to
confirm the download path and the pretrained checkpoint URL still work.

---

## Design decisions worth knowing about

These are the points where a merely-working submission and a good one diverge.
Each is implemented and commented in the solution; the skeleton poses them as
open questions rather than answering them.

**1. Frozen BatchNorm must be put in `eval()` mode.**
`requires_grad = False` does not stop a BN layer in training mode from updating
`running_mean` / `running_var` from CIFAR-10 batches, which progressively
corrupts the frozen ImageNet features. Symptom: validation accuracy plateaus
3–6 points low with no error message. This is the single most common silent bug
in submissions. See `set_frozen_bn_eval` and the `TransferModel.train` override.

**2. FRN placement.**
FRN normalizes over (H, W). Applied *after* global average pooling the spatial
size is 1×1, `x · rsqrt(mean(x²))` collapses to `sign(x)`, and all magnitude
information is destroyed. The reference applies it to the 1280×7×7 feature map
*before* pooling. A student who placed FRN after the pool and concludes "FRN
performed poorly" has measured an artefact of their own architecture, not a
property of FRN — grade the reasoning, and flag the placement.

**3. FRN is paired with TLU.**
Because FRN does not subtract the mean, activations are not centred, and a
fixed-threshold ReLU clips an arbitrary fraction of them. The paper introduces
TLU (`max(x, τ)`, τ learnable per channel) for exactly this reason and reports
that FRN without TLU underperforms BatchNorm. Students who use FRN + plain ReLU
should be expected to notice and discuss the resulting weakness.

**4. Filter-normalised landscape directions.**
BatchNorm and FRN make the loss invariant to rescaling of the weights, so a
model with larger weights looks spuriously flatter under raw random directions.
Cross-head landscape comparison is only meaningful with the per-filter rescaling
of Li et al. §4. Without it, the Step 8.3 comparison is noise.

**5. Clip-rate logging.**
`torch.nn.utils.clip_grad_norm_` returns the **pre-clipping** norm. The reference
logs it along with the fraction of steps that exceeded the threshold. This
converts "did clipping help?" from an accuracy-difference guess into a direct
measurement.

**6. Controlled comparison.**
Identical seed and head initialisation across all six runs; discriminative
learning rates (head 1e-3, unfrozen trunk 1e-4); weight decay excluded from
biases and from γ/β/τ, since the three heads have different numbers of such
parameters and decaying a normalizer's scale toward zero would penalise them
unequally; one fixed set of batches reused at every landscape grid point so the
surface reflects the model rather than sampling noise.

**7. Quantitative smoothness.**
`sharpness` = mean absolute discrete Laplacian of the loss surface, reported in
the summary table and in every landscape title, so Step 8.3 can be argued with a
number instead of by eyeballing contours.

**8. On-GPU resizing.**
`F.interpolate` inside `TransferModel.forward` rather than a `Resize` transform
in the dataloader — 32 → 224 is 49× the pixel volume, and doing it CPU-side
starves the GPU. Worth ~3–5× wall-clock on Colab.

---

## Rubric mapping

| Criterion | Weight | What to look at |
|---|---|---|
| Implementation correctness | 30% | FRN and TLU written from scratch (no built-in normalizer); freezing policy + frozen-BN handling; gradient hooks capturing pre-clip gradients; forward/backward sanity checks |
| Experimental design | 20% | Seeding discipline across runs; sensible LR split between head and trunk; all six configurations present; fixed evaluation batches; stated compute configuration |
| Analysis depth | 30% | Summary table beyond accuracy (generalisation gap, convergence speed, clip rate, gradient statistics, landscape smoothness); figures interpreted, not just produced |
| Report quality | 20% | Design choices defended; negative results reported honestly; limitations acknowledged |

Suggested partial-credit anchors for the 30% implementation block: data +
freezing 6, three heads incl. from-scratch FRN 9, gradient tracking + clipping 8,
landscape machinery 7.

---

## Expected findings

Grade on whether the evidence supports the claim, not on whether it matches this
list. These are the patterns that reproduce with the default hyperparameters.

- **All three heads land within a point or two of each other.** The frozen
  ImageNet trunk dominates. "The normalizer matters less than expected when most
  of the network is frozen" is the correct headline, and a student who reports a
  dramatic difference should be asked to check their seeds.
- **BatchNorm converges fastest** over the first 2–3 epochs and shows the largest
  early train/val discrepancy, since training uses batch statistics and
  evaluation uses running estimates.
- **LayerNorm is the most stable** — smoothest step-to-step gradient-norm trace,
  no train/eval discrepancy, batch-size independent. This is the one to
  recommend below batch size ~16.
- **FRN is competitive but more LR-sensitive**, and its extra parameters (γ, β,
  τ) show visibly larger gradient norms in the layer-wise plot.
- **Gradient clipping at 1.0 mostly binds during the first epoch** and rarely
  after. Its clearest effect is on the *variance* of the gradient-norm trace, not
  on final accuracy. "Clipping barely changed accuracy, and here is the clip-rate
  evidence" is the right answer, not a failed experiment.
- **Head landscapes are smooth and near-convex** within α, β ∈ [−1, 1] —
  unsurprising for a small head on frozen features. A chaotic landscape usually
  means the directions were not filter-normalised, or θ\* was not restored.

---

## Common failure modes in submissions

| Symptom | Cause | How to grade |
|---|---|---|
| Val accuracy plateaus 3–6 pts low, no error | Frozen BN left in train mode | Deduct under correctness; very common, consider a partial-credit note |
| "Colab is too slow / impossible" | 32→224 resize done in the dataloader | Not a correctness failure, but flag it; check they still ran all six configs |
| "FRN performs badly" | FRN placed after global pooling | Deduct under correctness; give credit if they diagnosed it themselves |
| Landscape looks like static | Directions not filter-normalised, or fixed batches not cached | Deduct under analysis depth |
| Every result after the landscape cell is wrong | θ\* not restored after the sweep | Correctness; usually visible as a sudden accuracy collapse mid-notebook |
| Gradient histogram is one bar at zero | Linear axis instead of log | Minor, deduct under visualisation quality |
| Large claimed gap between normalizers | Different seeds / head init per run | Deduct under experimental design — this is exactly the 20% criterion |
| Normalised with CIFAR-10 stats on ImageNet weights | Unexamined default | Acceptable **if justified in the report**; otherwise a design-control deduction |

---

## Limitations of this reference

Be prepared for students to raise these; credit them if they do.

- Single seed. A proper study would report mean ± std over 3–5 seeds; the
  observed 1–2 point differences are within plausible seed variance.
- One dataset, one architecture, one batch size — and batch size is precisely the
  variable that most differentiates BatchNorm from the other two. A batch-size
  sweep would be the single most informative extension.
- The landscape is a 2-D random slice through the head's parameter subspace, not
  the full geometry.
- No hyperparameter search per normalizer; all three share one LR, which mildly
  favours whichever normalizer happens to suit it.
- Dropout is active in the head, so the landscape is computed in `eval()` mode
  and reflects the deterministic network only.

---
