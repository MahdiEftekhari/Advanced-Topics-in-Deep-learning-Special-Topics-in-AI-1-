# DL4032 — Homework 01
## Transfer Learning Optimization: Normalization Techniques and Gradient Dynamics

**Instructor:** Dr. Mahdi Eftekhari · Shahid Bahonar University of Kerman
**Deadline:** three weeks from the assignment date

You will fine-tune a pre-trained MobileNetV2 on CIFAR-10 using three different
normalization schemes in the adaptation head — BatchNorm, LayerNorm, and Filter
Response Normalization — with and without gradient clipping, and analyse how
those choices shape gradient flow and the loss landscape.

Read `DL4032_HW01.pdf` first. This file explains how to actually get started.

---

## Contents

```
TransferLearning/
├── DL4032_HW01.pdf            the assignment specification (authoritative)
├── dl4032_hw01_skeleton.py    your starting point — fill in every TODO
└── README.md                  this file
```

The skeleton contains every class and function signature required by the
handout, in the same order as the nine steps. **Do not rename or change the
signatures** — the grading scripts call them by name. You may add as many
helper functions as you like.

---

## Environment setup

### Google Colab (recommended)

```python
!nvidia-smi                       # confirm you actually have a GPU attached
!pip install -q torch torchvision matplotlib numpy pandas
```

Runtime → Change runtime type → **T4 GPU**. Everything in this assignment fits
in a free-tier T4 if you follow the compute budget below.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib numpy pandas
python dl4032_hw01_skeleton.py --smoke
```

CPU-only will work for the smoke run but not for the full sweep.

---

## How to work through it

The skeleton is organised so each section is independently testable. Do them in
order and do not move on until the checkpoint passes.

| Step | Sections | Checkpoint before moving on |
|---|---|---|
| 1 — Data | 1.1–1.4 | Loaders yield `(N, 3, 32, 32)` batches; your sample grid shows recognisable images (not grey mush) |
| 2 — Base model | 2.1–2.4 | A random batch produces `(N, 10)`; gradients reach the head but **no** frozen parameter has a gradient |
| 3 — Norm variants | 3.1–3.4 | All three heads run on the same batch without shape errors |
| 4 — Gradients | 4.1–4.4 | Gradient norms are non-zero and change between steps |
| 5 — Training | 5.1–5.4 | One epoch on a 512-image subset completes and logs sensible numbers |
| 6 — Landscapes | 6.1–6.4 | The surface is finite everywhere, and model weights are unchanged after the sweep |
| 7 — Experiments | — | All six configurations run to completion with logs saved |
| 8 — Analysis | — | Every figure has axis labels, a legend, and a caption you actually wrote |
| 9 — Report | — | 5–8 page PDF |

Run `python dl4032_hw01_skeleton.py --smoke` constantly. It uses a 512-image
subset at 96×96 for one epoch, so it exercises the whole pipeline in a couple of
minutes. **Do not launch the full 6 × 15-epoch sweep until the smoke run is
clean end to end.** Debugging a crash that happens in hour four is not a good
use of your three weeks.

---

## Compute budget — read this before you plan your time

The full sweep at 224×224 over the whole training split is roughly **6–9 minutes
per epoch on a T4**, i.e. 10–13 GPU-hours for all six experiments. That does not
fit in one Colab session, and Colab will disconnect on you.

Three workable strategies:

1. **Reduced preset (recommended).** 160×160 input, 10 000 training images:
   about 90 minutes total. The differences between normalizers are still clearly
   visible, and the assignment explicitly values analysis over peak accuracy.
2. **Two runs per session.** Full resolution, but run two configurations at a
   time, save checkpoints and JSON logs to Google Drive, and reload them for the
   analysis step without retraining.
3. **Full sweep** only if you have access to a persistent GPU.

Whichever you choose, state it in your report's experimental setup section. A
reduced configuration honestly described costs you nothing; an undocumented one
makes your results unreproducible.

Mount Drive and save there, not to the ephemeral Colab disk:

```python
from google.colab import drive
drive.mount('/content/drive')
# then pass --save-dir /content/drive/MyDrive/dl4032_hw01
```

---

## Things that reliably go wrong

Not answers — just the places where people lose days.

- **Resizing in the dataloader.** Going 32 → 224 inside the `transform` moves 49×
  the pixel data through the CPU every batch, and the GPU sits idle. Think about
  where that resize is cheapest.
- **Frozen BatchNorm.** Setting `requires_grad = False` does not stop a
  BatchNorm layer in training mode from doing something else it does. Find out
  what, and decide whether you want it.
- **FRN's spatial axes.** FRN normalizes over height and width. Check what those
  dimensions are at the point in your head where you placed it.
- **Gradient histograms on a linear axis.** Gradients span several orders of
  magnitude; a linear histogram is one spike at zero.
- **Not restoring the weights after the landscape sweep.** If you perturb
  parameters in place and forget to put them back, every later cell in your
  notebook is silently evaluating a broken model.
- **Different seeds across runs.** If the head is initialised differently for
  your BatchNorm and LayerNorm runs, part of the gap you report is initialisation
  noise, not normalization. Controlling confounders is 20% of the grade.
- **Clipping that never fires.** Log how often the clip threshold is actually
  exceeded. If it is near zero, clipping did nothing — that is a real finding and
  you should report it as one, with the evidence.

---

## Deliverables

1. **Code** implementing all components (`.py` or a cleaned `.ipynb`).
2. **Checkpoints** for each of the six experiments.
3. **Visualization notebook** containing every plot with your interpretation
   written next to it — not a wall of figures at the end.
4. **Final report**, PDF, 5–8 pages.
5. **Slides** summarising the findings — optional, extra points.

Suggested report structure:

1. Introduction — transfer learning; what each of the three normalizers is
   invariant to (batch / feature / spatial axis) and why that might matter here.
2. Method — architecture, freezing policy, where you placed FRN and why, learning
   rates, and what you held fixed across the six runs.
3. Results — summary table, training curves, gradient statistics.
4. Loss landscapes — how you generated the directions, side-by-side comparison,
   and whether landscape smoothness tracks the generalisation gap.
5. Discussion — when would each normalizer matter more than it did here?
6. Limitations — single seed, one dataset, most of the trunk frozen, and a 2-D
   slice through a high-dimensional surface.
7. Conclusions and practical recommendations.

Submit as `LastName_FirstName_StudentID_HW01.zip`.

---

## Grading

| Criterion | Weight |
|---|---|
| Implementation correctness | 30% |
| Experimental design | 20% |
| Analysis depth | 30% |
| Report quality | 20% |

A working implementation with a shallow write-up scores lower than a modest
implementation with sharp analysis. If your six runs end up within a point or
two of each other, that is a legitimate result — explain *why* rather than
hunting for a difference that isn't there.

---

## References

- Singh & Krishnan (2020), *Filter Response Normalization Layer: Eliminating
  Batch Dependence in the Training of Deep Neural Networks* — read §3 carefully
  before implementing FRN, and note what the paper pairs it with.
- Li et al. (2018), *Visualizing the Loss Landscape of Neural Nets* — §4 explains
  why raw random directions are not enough.
- He et al. (2015), *Delving Deep into Rectifiers*.
- [MobileNetV2 in torchvision](https://pytorch.org/vision/stable/models/mobilenetv2.html)
- [PyTorch normalization layers](https://pytorch.org/docs/stable/nn.html#normalization-layers)
- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

Questions go to the course forum, not to individual email — if you hit
something, someone else has too.

Good luck.