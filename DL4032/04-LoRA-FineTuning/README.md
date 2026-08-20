# Deep Learning — Final Assignment
## Parameter-Efficient Fine-tuning with LoRA

**Instructor:** Dr. Mahdi Eftekhari
**Points:** 100 · **Estimated time:** 15–20 hours

You will adapt GPT-2 to the SQuAD question-answering dataset using Low-Rank
Adaptation (LoRA), then evaluate it with SQuAD F1 / Exact Match and analyse the
trade-offs of parameter-efficient fine-tuning.

Read `GPT2_fine-tune.pdf` first — it is the authoritative specification. This
file explains how to get started and where people lose time.

---

## Contents

```
LoRA_FineTuning/
├── GPT2_fine-tune.pdf              assignment specification (authoritative)
├── GPT2_instruction_tuning.py      your starting point — complete every TODO
├── requirements.txt                pinned dependencies
└── README.md                       this file
```

The skeleton contains every function and class the handout requires, in order.
**Keep the public names and signatures** (`preprocess_function`,
`QADataCollator`, `generate_answer`) — the grading scripts import them by name.
Add whatever helpers you like.

---

## Environment setup

### Google Colab (recommended)

```python
!nvidia-smi                     # confirm a GPU is attached
!pip install -q -r requirements.txt
```

Runtime → Change runtime type → **T4 GPU**. GPT-2 small with LoRA fits
comfortably in free-tier memory.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Version sensitivity.** The `transformers` Trainer API changed in v5. Two things
that break silently if you copy code from older tutorials or blog posts:

| Old (≤ v4) | Current (v5) |
|---|---|
| `Trainer(tokenizer=...)` | `Trainer(processing_class=...)` |
| `TrainingArguments(evaluation_strategy=...)` | `TrainingArguments(eval_strategy=...)` |

`requirements.txt` pins a known-good set. If you deviate, check
`inspect.signature(Trainer.__init__)` before assuming a keyword exists.

---

## How to work through it

Do the tasks in order and verify each before moving on. Every stage is
independently testable and later bugs are far harder to localise.

| Task | Points | Checkpoint before moving on |
|---|---|---|
| 1 — Preprocessing | 25 | Decode one processed example and confirm by eye that every position with `label != -100` falls inside the answer span, and nowhere else |
| 2 — Data collation | 20 | A batch of unequal-length rows produces correctly shaped tensors; running the collator twice on the same row gives identical output |
| 3 — LoRA + training | 30 | `print_trainable_parameters()` reports well under 1% trainable; loss decreases over the first few hundred steps |
| 4 — Inference + eval | 25 | `generate_answer` returns only the answer, with no echoed prompt and no run-on continuation |

### Verifying label masking (do this before you train anything)

The single highest-value ten minutes in this assignment:

```python
ex = tok_train_ds[0]
for tid, lab in zip(ex['input_ids'], ex['labels']):
    if lab != -100:
        print(repr(tokenizer.decode([tid])), end=' ')
```

If that prints anything from the context or question, your masking is
misaligned and every gradient you compute afterwards is wrong. The handout lists
"Label Misalignment" as a common pitfall for a reason.

Two things that make this subtle: tokenizers are not additive, so
`len(tokenize(prompt))` is not always the number of prompt tokens inside
`tokenize(prompt + answer)`; and if you pad to a fixed length, the padding
positions need masking too or the model is trained to emit endless pad tokens.

---

## Compute budget

GPT-2 small + LoRA on 2 000 SQuAD examples is roughly **2–4 minutes per epoch**
on a T4. Ten epochs is well under an hour, so the base run is cheap.

The cost is in the **ablations**, which are where 30% of your grade lives. Task 3
asks for four ranks × three learning rates × two target-module sets, and Task 4
asks for four decoding-strategy families. Do not run the full cross-product —
that is 24 training runs. Vary one axis at a time from a fixed baseline
(~9 runs), and say so explicitly in your report. A well-justified one-factor
sweep scores better than an unfinished grid.

Save to Drive, not to Colab's ephemeral disk:

```python
from google.colab import drive; drive.mount('/content/drive')
# output_dir="/content/drive/MyDrive/lora_qa/results"
```

Decoding-strategy comparisons need **no retraining** — generate from one
checkpoint with different `generate()` arguments. Budget accordingly.

---

## Things that reliably go wrong

Not answers, just where the time goes.

- **A collator you build but never use.** `Trainer` accepts a `data_collator`
  argument. If you don't pass it, your class is dead code and `Trainer` silently
  falls back to its default. Confirm yours actually runs — put a `print` or a
  counter in `__call__` for one step.
- **In-place mutation of dataset rows.** If you pad with `list.extend()` on the
  list you got from the dataset, you are modifying the cached row, not a copy.
  With a persistent cache this corrupts data across epochs. Copy first.
- **Static vs dynamic padding.** Task 2 explicitly asks for padding to the *batch*
  maximum. Padding every batch to a fixed 256 is simpler but wastes most of each
  batch when sequences are short — measure the waste and discuss it.
- **Evaluating the wrong model.** After `save_pretrained`, reloading with
  `AutoPeftModelForCausalLM` gives you the adapter on the base model. Verify the
  adapter is really attached, and that the model is in `eval()` mode.
- **Prompt echo in generations.** Slicing the decoded string by `len(prompt)`
  breaks when special tokens are skipped or whitespace changes. Slice by
  **token count** on the generated ids instead.
- **No baseline.** Task 4 requires comparison against the non-fine-tuned model.
  Run it — a large fraction of your F1 may come from prompt format alone, and
  that is a finding worth reporting.
- **Evaluating on five examples.** The skeleton's `vali_ds` is 5 rows, useful
  only as a smoke test. SQuAD F1 over 5 examples has an enormous confidence
  interval, and the report asks for confidence intervals. Use several hundred
  held-out examples for any number you intend to defend.

---

## Deliverables

Per the handout, submit:

```
assignment_submission/
├── lora_qa_finetuning.py
├── report.pdf
├── results/
│   ├── model_checkpoints/
│   ├── evaluation_results.json
│   └── plots/
├── requirements.txt
└── README.md
```

1. **Implementation (40 pts)** — runs without errors, documented, handles edge
   cases.
2. **Report (40 pts)** — 6–8 pages, IEEE conference style: methodology,
   results with confidence intervals and ablations, analysis and failure cases.
3. **Theoretical questions (20 pts)** — LoRA forward/backward derivation,
   why prompt masking matters, and complexity comparison with full fine-tuning.
   Answer with mathematical rigour, not prose summaries.

On the theory section: the derivation should show gradients flowing through
both `A` and `B`, and should explain why `W` receiving no gradient is what makes
LoRA cheap. For the complexity question, compute the actual reduction factor for
*your* configuration and report the real trainable-parameter count from
`print_trainable_parameters()`.

---

## Grading

| Criterion | Weight |
|---|---|
| Technical implementation | 40% |
| Experimental rigour | 30% |
| Theoretical understanding | 20% |
| Communication | 10% |

Working code with a thin report scores below modest code with sharp analysis.
Negative and null results, honestly documented with evidence, are worth full
credit — "rank 32 gave no measurable gain over rank 8, here is the table and the
overlapping confidence intervals" is a good result, not a failed experiment.

---

## References

- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*
- Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Reading
  Comprehension*
- Vaswani et al. (2017), *Attention Is All You Need*
- [Transformers docs](https://huggingface.co/docs/transformers/) ·
  [PEFT docs](https://huggingface.co/docs/peft/) ·
  [PyTorch docs](https://pytorch.org/docs/)

Questions go to the course forum rather than individual email — if you hit
something, someone else has too.

Good luck.
