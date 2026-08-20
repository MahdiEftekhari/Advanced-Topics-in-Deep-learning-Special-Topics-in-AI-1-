# LoRA Fine-tuning — Reference Solution

**Learning material, not a submission.** This is the reference solution and
answer key for the assignment, kept here for instructors/TAs grading it and for
students studying the topic after the fact. If you are currently working on
this assignment, use `../GPT2_instruction_tuning.py` and attempt it yourself
first — come back here to compare, and to check the theory answer key, only
once you've submitted your own solution. GitHub permissions are repo-wide, so
while the assignment is still open, keep student-facing files in a separate
repository rather than relying on this directory boundary to hide the answer
key.

---

## Contents

```
Solution/
├── lora_qa_finetuning.py   corrected reference implementation — USE THIS
├── final_code.py           original draft, kept for reference — DO NOT USE
└── README.md               this file
```

**`lora_qa_finetuning.py` is the answer key.** It fixes every defect listed
below and adds the experimental machinery the report deliverable depends on. It
is named to match §9 of the PDF and corresponds to
`../GPT2_instruction_tuning.py` section by section, so a submission can be graded
by scrolling the two side by side.

`final_code.py` is retained only so the defects can be pointed at concrete lines
when explaining them to a student or to next year's TA. Do not distribute it or
grade against it.

---

## Running it

```bash
pip install -r ../requirements.txt

python lora_qa_finetuning.py --mode selftest   # no GPU, no network — ~2 seconds
python lora_qa_finetuning.py --mode smoke      # ~5 min end-to-end on a T4
python lora_qa_finetuning.py --mode single     # one baseline run
python lora_qa_finetuning.py --mode all        # + ablations + decoding study
```

`--mode selftest` covers every component that needs no downloads: label-boundary
exactness, answer retention under long contexts, dropping of empty and over-long
answers, off-by-one detection in both directions, dynamic padding widths,
exclusion of padding from the loss, SQuAD normalisation and max-over-golds,
bootstrap intervals, and the paired significance test. Run it first whenever
something looks wrong — it localises the failure in seconds without a GPU.

Outputs under `--output-dir` (default `./results`): `model_checkpoints/<run>/`,
`runs/<run>.json`, `evaluation_results.json`, `plots/*.png`,
`failure_analysis.md`.

---

## Verification status

The self-test suite passes. Separately, the transformers/PEFT integration was
exercised against a tiny locally-constructed GPT-2 (2 layers, 32-dim, random
weights, no download), confirming that:

- LoRA attaches to GPT-2's `Conv1D` `c_attn` with only `lora_*` params trainable;
- the `attn+mlp` preset attaches strictly more adapters than `attn`;
- `Trainer.data_collator` **is** the custom collator, not a silently-installed default;
- training logs losses and evaluates without error;
- the loss is provably unchanged when padded positions are overwritten with junk;
- batched greedy generation is byte-identical at batch size 1 and 3, i.e. left padding is correct;
- evaluation, failure analysis, JSON serialisation and all four plot functions run.

**Not verified:** a real run on real GPT-2 weights and real SQuAD — the authoring
environment had no access to `huggingface.co`. Do one `--mode smoke` on the
university network before the semester starts.

---

## Coverage against the 100 points

| Deliverable | Pts | `lora_qa_finetuning.py` | `final_code.py` |
|---|---:|---|---|
| Task 1 — preprocessing | 25 | Exact boundary, answer-aware truncation, padding excluded from loss, runtime assertions | Defective (1, 2, 4) |
| Task 2 — collation | 20 | Dynamic padding, wired into `Trainer`, efficiency measured | Collator **never called**; static padding |
| Task 3 — LoRA + training | 30 | Rank / LR / target-module ablations, curves, params-vs-performance, cost | Single configuration |
| Task 4 — inference + eval | 25 | Held-out split, bootstrap CIs, zero-shot baseline, paired significance test, 11 decoding strategies, categorised failures | 5 examples drawn from `train`, one sampled strategy |
| Report | 40 | Produces every figure and number the report needs | Not covered |
| Theory | 20 | Answer key below | Not covered |

The reference produces the *artefacts*; the report prose remains the student's work.

---

## Defects in `final_code.py`

All fixed in `lora_qa_finetuning.py`, tagged `[FIX-n]` at the relevant code.
Several are exactly the pitfalls §7 of the PDF warns students about, which is
what made them worth documenting rather than quietly patching.

### 1. Padding tokens contribute to the loss — critical

Preprocessing tokenizes with `padding="max_length"`, then builds labels as
`input_ids.copy()` and masks only the prompt prefix. Since `pad_token =
eos_token`, every padded slot holds token 50256 **as a live training target**. A
256-token sequence with a 40-token prompt and a 3-token answer computes loss over
~213 padding targets against 3 real ones — the answer signal is outweighed by
roughly two orders of magnitude, and the model is trained to emit endless EOS.

*Fix:* mask everything outside the answer span; better, don't pad in
preprocessing at all and let the collator handle it.

### 2. Off-by-one at the prompt/answer boundary — critical

Prompt length is measured by tokenizing the prompt string in isolation:

```python
prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
```

GPT-2's BPE attaches a leading space to the following word. Tokenized alone the
trailing space is its own token; inside the joint sequence it merges into
`" Rome"`. So `prompt_length` overshoots by one and the mask consumes the **first
token of the answer**. For the one- and two-token answers that dominate SQuAD,
that removes most of the supervision while still producing a plausible loss curve.

*Fix:* end the prompt at `"Answer:"`, give the answer its own leading space, and
build the sequence by concatenating separately tokenized id lists — the boundary
is then exact by construction. `verify_alignment()` asserts it at runtime and
rejects an off-by-one in either direction.

### 3. The custom collator is never used — high

`data_collator = QADataCollator(...)` is built but not passed to `Trainer(...)`.
Because a tokenizer is supplied, `Trainer` installs its own default collator
instead. The pipeline only works at all because preprocessing already padded
everything to exactly 256, so the default has nothing to do. This silently zeroes
out Task 2 (20 points) while appearing to satisfy it.

`QADataCollator` also pads to a fixed `max_length`, whereas §4.2 requires
**dynamic padding to batch maximum length**. Both need fixing.

### 4. Truncated answers are not handled — high

`max_length=256` with `truncation=True` silently discards the answer whenever the
context is long, which is common in SQuAD. Such examples carry labels that are
entirely `-100` and contribute `NaN`. §4.1 asks students explicitly how they
handle this; the original does not.

*Fix:* truncate the *context* to a window centred on the answer (using
`answer_start`), so the answer is always present; drop and count what still
doesn't fit.

### 5. Evaluation set is five examples drawn from `train` — high

`vali_ds = dataset['train'].select(range(5))` is taken *before* the split, so
those rows come from the same pool the model trains on and may literally appear
in `train_ds`. Five examples cannot support the confidence intervals §5.2
requires. SQuAD's own `validation` split is loaded and never used.

### 6. The collator mutates dataset rows in place — medium

```python
input_ids = feature['input_ids']
input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
```

`input_ids` is a reference, not a copy, so `.extend` mutates the caller's list.
HF datasets materialise a fresh list per access so this is usually invisible, but
against any in-memory or cached list-backed dataset it corrupts rows across
epochs — a genuinely nasty bug to diagnose. *Fix:* `list(feature['input_ids'])`.

### 7. Non-deterministic decoding for a deterministic metric — medium

`do_sample=True, temperature=0.7` makes EM/F1 change between runs. Greedy is the
sane default for extractive QA; sampling belongs in the §4.4 decoding comparison,
reported as such.

### 8. Brittle answer extraction and no stopping rule — medium

`generated_text[len(prompt):]` slices the *decoded* string by the *original*
prompt's character length. With `truncation=True, max_length=200` the decoded
prompt can be shorter, and `skip_special_tokens=True` shifts it further,
producing empty or corrupted answers. Slice by token count instead. Separately,
GPT-2 rarely emits EOS, so generation runs on into a fabricated next
Context/Question block; without a stopping rule the F1 precision denominator
explodes.

### 9. Device handling and reproducibility — medium

`AutoPeftModelForCausalLM.from_pretrained` returns a CPU model and
`generate_answer` never moves anything to CUDA. No global seed is set (only the
dataset shuffle), so nothing is reproducible run to run — which directly costs
points under the Reproducibility criterion the assignment itself lists.

### 10. Version fragility and minor issues — medium/low

The file mixes the *new* `eval_strategy` with the *removed* `Trainer(tokenizer=)`,
so it targets no single `transformers` release: `eval_strategy` needs ≥ 4.41,
`tokenizer=` was removed in 5.x. `../requirements.txt` pins `>=4.41,<5`; the
reference instead inspects the signatures at runtime and adapts.

Also: `targets = []` is built and never used; `DataCollatorForLanguageModeling`
and `AutoConfig` are imported/used redundantly; `fan_in_fan_out=False` is wrong
in principle for GPT-2's `Conv1D` layers (recent PEFT auto-detects and overrides
with a warning, so a student who reasons about it and sets `True` deserves
credit); and 10 epochs on 2000 examples overfits well before the end.

---

## Expected results

GPT-2 small (124M, no instruction tuning), LoRA r=8 on `c_attn`, 2000 SQuAD
examples. Expect **low absolute scores**: modest F1 driven largely by token
overlap, Exact Match close to zero. Fine-tuning mainly teaches answer *format* —
emit a short span and stop — rather than comprehension.

Grade accordingly. A student reporting weak numbers with a clean baseline
comparison and honest failure analysis has done the assignment correctly. A
student reporting strong F1 should be checked for evaluating on training
examples, or for scoring the full generated string including the echoed prompt.

Useful sanity check: with defects 1 and 2 fixed, results improve noticeably. If a
student's numbers are indistinguishable from the zero-shot baseline, their
preprocessing is the first place to look.

---

## Grading guidance

| Component | Pts | What to look for |
|---|---:|---|
| Task 1 | 25 | Boundary verified explicitly — most students never look; padding masked out of the loss; truncation policy stated and justified |
| Task 2 | 20 | Collator actually wired into `Trainer`; padding genuinely dynamic; static-vs-dynamic analysis engages with memory, not just correctness |
| Task 3 | 30 | Ablations on all three axes; one axis varied at a time from a stated baseline; seeds fixed; parameter count vs. performance plotted |
| Task 4 | 25 | Evaluation set large enough for intervals; zero-shot baseline present; decoding strategies compared; failure cases quoted, not just counted |

Suggested deductions: evaluating on training examples (−5, under Experimental
Rigor rather than Implementation); ablations at a single seed without
acknowledgement (−3); custom collator unused (−10 of Task 2's 20).

Give credit for evidenced negative results. "Rank 32 did not beat rank 8, here
are the curves and the parameter counts" is a better answer than an unexplained
claim that it did.

---

## Theory answer key (§5.3, 20 points)

### Q1 — LoRA forward and backward (8 pts)

For a frozen `W ∈ R^{d×k}` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `r ≪ min(d,k)`:

```
h = Wx + (α/r)·BAx
```

With `g = ∂L/∂h` and `z = Ax`:

```
∂L/∂B = (α/r)·g·zᵀ              (d×r)
∂L/∂A = (α/r)·Bᵀg·xᵀ            (r×k)
∂L/∂x = Wᵀg + (α/r)·AᵀBᵀg
```

Require for full marks: `A` is initialised Gaussian and `B` at zero, so `ΔW = 0`
at step 0 and the adapted model starts exactly at the pre-trained function; and
`∂L/∂W` is never formed, but `Wᵀg` **is** still computed, so backward through the
frozen weights is unavoidable. The saving is in optimizer state and gradient
storage, not in backward FLOPs. The common error is claiming LoRA makes the
backward pass computationally cheap.

### Q2 — Loss masking (6 pts)

Masking restricts the objective to `L = −Σ_{t=T_prompt}^{T_total} log P(x_t|x_{<t})`,
so gradients flow only from answer tokens. Without it: (a) the objective is
dominated by the context, typically 10–50× longer than the answer, so capacity
goes into reproducing Wikipedia prose; (b) the model is trained to *generate*
questions and contexts, which is not the inference-time distribution, where the
prompt is always given; (c) with `pad_token = eos_token`, unmasked padding trains
the model to emit endless EOS — defect 1 above, in its natural habitat.

Note that prompt tokens still participate as **conditioning** via the attention
mask; masking removes them from the loss, not from the input. Students who
conflate the two have the wrong mental model and should lose points even if the
code happens to work.

### Q3 — Complexity and parameter reduction (6 pts)

For this configuration — GPT-2 small, 12 layers, `d = 768`, `c_attn ∈ R^{768×2304}`,
`r = 8`, attention only:

- per layer: `A` is 8×768 = 6,144 and `B` is 2304×8 = 18,432 → **24,576**
- total trainable: 24,576 × 12 = **294,912**
- base 124,439,808 → total 124,734,720
- trainable fraction **0.2364%**, a **≈422×** reduction

Adam memory (fp32 params + two moments + gradient ≈ 16 bytes/param): ~2.0 GB for
full fine-tuning versus ~4.7 MB for LoRA. Forward FLOPs rise slightly (the extra
`BAx`); backward FLOPs are essentially unchanged, per Q1. Per-task storage is a
~1.2 MB adapter rather than a ~500 MB checkpoint — the practical argument at
serving time.

Accept any configuration-consistent arithmetic, but require that the student
computes it for **their own** rank and target modules. Adding the MLP projections
roughly triples the trainable count.

---

## Limitations of the reference

Credit students who raise these.

- Single seed. A proper study reports mean ± std over 3–5 seeds; the differences
  between ablation points are plausibly within seed variance.
- One dataset, one architecture, one batch size.
- Ablations are one-factor-at-a-time, so interactions (e.g. high rank *with* high
  LR) are not measured. This is a deliberate compute trade-off, stated in the code.
- At inference the answer position is unknown, so the context window is taken
  from the start rather than centred on the answer as in training. When the
  answer falls outside that window the example is unanswerable by construction.
  This train/inference asymmetry is inherent to the prompt format, not a bug, and
  is worth a paragraph in a strong report.
- `lora_alpha` is scaled with `r` in the rank sweep so that α/r stays constant.
  Without this the rank ablation would silently vary update magnitude too.
