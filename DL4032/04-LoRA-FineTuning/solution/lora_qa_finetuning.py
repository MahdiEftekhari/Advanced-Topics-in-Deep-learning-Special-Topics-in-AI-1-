#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Final Assignment -- Parameter-Efficient Fine-tuning with LoRA
GPT-2 adapted to extractive question answering on SQuAD.
==============================================================================
REFERENCE SOLUTION  (instructor / TA copy -- not for distribution to students)

Course : Deep Learning, Dr. Mahdi Eftekhari

Covers every graded component of the handout:

    Task 1  Data preprocessing            (25 pts)  -> section 2
    Task 2  Custom data collation         (20 pts)  -> section 3
    Task 3  LoRA configuration + training (30 pts)  -> sections 4, 5, 9
    Task 4  Inference and evaluation      (25 pts)  -> sections 6, 7, 8
    Report  ablations, plots, baseline,
            significance testing                    -> sections 9, 10

Points where this implementation deliberately departs from the obvious approach
are marked [FIX-n] and explained in full in Solution/README.md.

Usage
-----
    python lora_qa_finetuning.py --mode selftest    # no GPU, no network needed
    python lora_qa_finetuning.py --mode smoke       # ~5 min end-to-end check
    python lora_qa_finetuning.py --mode single      # one baseline run
    python lora_qa_finetuning.py --mode all         # everything, in order

Outputs (under --output-dir, default ./results):
    model_checkpoints/<run>/     LoRA adapter weights (~1.2 MB each)
    runs/<run>.json              per-run config, curves, cost, metrics
    evaluation_results.json      everything, in one file
    plots/*.png                  curves, ablations, cost, decoding study
    failure_analysis.md          categorised failures with examples
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Heavy imports are deferred so `--mode selftest` runs in a bare environment.
try:
    import torch
    _HAS_TORCH = True
except ImportError:                                       # pragma: no cover
    _HAS_TORCH = False


# =============================================================================
# 1. Reproducibility, device, library-version compatibility
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    """[FIX-9] Seed everything.

    The ablation study compares configurations differing in one factor. If the
    seed varies too, the comparison is confounded and the "Reproducibility"
    criterion cannot be met. The original code seeded only dataset shuffling.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def get_device():
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_training_arguments(**kwargs):
    """[FIX-9b] Build TrainingArguments across transformers 4.4x and 5.x.

    `evaluation_strategy` was renamed `eval_strategy` in 4.41 and the old name
    was removed in 5.x; passing the wrong one raises. Filtering against the real
    signature keeps the file working on whatever the student or lab has pinned.
    """
    from transformers import TrainingArguments
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    for key in [k for k in kwargs if k not in params]:
        kwargs.pop(key)
    return TrainingArguments(**kwargs)


def make_trainer(model, args, train_dataset, eval_dataset, data_collator, tokenizer):
    """[FIX-3] Pass `data_collator` EXPLICITLY, and handle the v5 rename of
    `tokenizer=` to `processing_class=`.

    Omitting `data_collator` while supplying a tokenizer makes Trainer install
    its own default collator, so a custom one is silently never called.
    """
    from transformers import Trainer
    params = inspect.signature(Trainer.__init__).parameters
    kw: Dict[str, Any] = dict(model=model, args=args, train_dataset=train_dataset,
                              eval_dataset=eval_dataset, data_collator=data_collator)
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    return Trainer(**kw)


# =============================================================================
# 2. Task 1 -- Configuration, data loading, preprocessing
# =============================================================================

@dataclass
class RunConfig:
    """One training run. Everything that can affect a number in the report lives
    here and is serialised beside the metrics, so any result is traceable."""
    name: str = "baseline"
    model_name: str = "gpt2"

    max_length: int = 384
    n_train: int = 2000
    n_eval: int = 200          # in-training validation (loss only)
    n_test: int = 300          # held-out SQuAD validation split, for EM/F1

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("c_attn",)

    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "cosine"
    seed: int = 42

    decoding: str = "greedy"
    max_new_tokens: int = 24

    def tag(self) -> str:
        mods = "+".join(self.target_modules)
        return f"{self.name}_r{self.lora_r}_lr{self.learning_rate:g}_{mods}_s{self.seed}"

    def replace(self, **kwargs) -> "RunConfig":
        data = asdict(self)
        data["target_modules"] = tuple(data["target_modules"])
        data.update(kwargs)
        return RunConfig(**data)


# GPT-2 note: `c_proj` names BOTH the attention output projection and the MLP
# output projection, so "attn+mlp" adapts all four projections per block.
TARGET_PRESETS: Dict[str, Tuple[str, ...]] = {
    "attn": ("c_attn",),
    "attn+mlp": ("c_attn", "c_proj", "c_fc"),
}

PROMPT_HEAD = "Context:"
PROMPT_TAIL = "\nQuestion: {question}\nAnswer:"


def load_squad_splits(cfg: RunConfig):
    """Three disjoint splits.

    [FIX-5] The original took its final evaluation examples from `train` before
    splitting, and used five of them. Here train/eval are disjoint subsets of
    `train`, while everything reported as EM/F1 comes from SQuAD's own
    `validation` split, which the model never sees -- and is large enough to
    support bootstrap intervals.
    """
    from datasets import load_dataset

    raw = load_dataset("squad")
    split = raw["train"].train_test_split(test_size=0.1, seed=cfg.seed)
    train_ds = split["train"].shuffle(seed=cfg.seed).select(
        range(min(cfg.n_train, len(split["train"]))))
    eval_ds = split["test"].shuffle(seed=cfg.seed).select(
        range(min(cfg.n_eval, len(split["test"]))))
    test_ds = raw["validation"].shuffle(seed=cfg.seed).select(
        range(min(cfg.n_test, len(raw["validation"]))))
    del raw, split
    gc.collect()
    return train_ds, eval_ds, test_ds


def _ids(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_prompt_ids(tokenizer, context: str, question: str, budget: int,
                     answer_start: Optional[int] = None) -> List[int]:
    """Assemble prompt token ids, truncating the CONTEXT rather than the sequence.

    [FIX-4] Truncating the joint sequence from the right at a fixed length
    silently deletes the answer whenever the context is long -- common in SQuAD.
    Those examples end up with labels that are entirely -100, which produces NaN
    for their loss contribution. Here the context is reduced to a window centred
    on the answer (training, where `answer_start` is known), so the answer is
    always present. At inference `answer_start` is unknown, so the leading
    `budget` tokens are kept instead; that asymmetry is a real limitation and is
    quantified in the report rather than hidden.
    """
    head_ids = _ids(tokenizer, PROMPT_HEAD)
    tail_ids = _ids(tokenizer, PROMPT_TAIL.format(question=question.strip()))
    ctx_budget = budget - len(head_ids) - len(tail_ids)
    if ctx_budget <= 0:
        return []

    ctx_ids = _ids(tokenizer, " " + context.strip())
    if len(ctx_ids) <= ctx_budget:
        return head_ids + ctx_ids + tail_ids
    if answer_start is None:
        return head_ids + ctx_ids[:ctx_budget] + tail_ids

    # Map the answer's character offset to a token offset via the prefix length.
    prefix_len = len(_ids(tokenizer, " " + context[:answer_start].strip()))
    start = max(0, prefix_len - ctx_budget // 2)
    start = min(start, len(ctx_ids) - ctx_budget)
    return head_ids + ctx_ids[start:start + ctx_budget] + tail_ids


def build_example(tokenizer, context: str, question: str, answer_text: str,
                  answer_start: Optional[int], max_length: int) -> Optional[Dict]:
    """One training example with an exactly aligned label mask.

    [FIX-2] The prompt/answer boundary is established by CONCATENATING
    SEPARATELY TOKENIZED id lists -- never by tokenizing the prompt alone,
    measuring its length, and slicing the joint encoding at that index.

    GPT-2's byte-pair tokenizer attaches a leading space to the following word.
    `"Answer: "` tokenized on its own ends in a lone-space token, but inside
    `"Answer: Rome"` that space merges into a single `" Rome"` token. Measuring
    the prompt separately therefore overshoots by one and masks the FIRST TOKEN
    OF THE ANSWER. Since SQuAD answers are frequently one or two tokens, that
    removes most of the supervision while still producing a plausible loss curve.

    The prompt here ends at `"Answer:"` and the answer carries its own leading
    space, so the two pieces tokenize independently and the boundary is exact by
    construction.

    [FIX-1] Labels are -100 everywhere except the answer span and its EOS. No
    padding is added at this stage at all -- padding belongs to the collator --
    so padded positions can never leak into the loss. The original copied
    `input_ids` (already padded to a fixed width with `pad_token == eos_token`)
    into the labels, which trained the model on ~200 padding targets per example
    against 3 real ones.
    """
    answer_text = (answer_text or "").strip()
    if not answer_text:
        return None

    answer_ids = _ids(tokenizer, " " + answer_text) + [tokenizer.eos_token_id]
    if len(answer_ids) >= max_length:
        return None

    prompt_ids = build_prompt_ids(tokenizer, context, question,
                                  budget=max_length - len(answer_ids),
                                  answer_start=answer_start)
    if not prompt_ids:
        return None

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    assert len(input_ids) == len(labels) <= max_length
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "n_prompt": len(prompt_ids),
            "n_answer": len(answer_ids)}


def make_preprocess_fn(tokenizer, max_length: int):
    """Batched `map` function.

    Examples that cannot be represented are dropped -- a batched map may return
    fewer rows than it received -- and counted, so the report can state exactly
    how much data was discarded and why.
    """
    stats = {"kept": 0, "dropped": 0}

    def preprocess_function(examples: Dict[str, List]) -> Dict[str, List]:
        out: Dict[str, List] = {"input_ids": [], "attention_mask": [], "labels": []}
        for context, question, answers in zip(examples["context"],
                                              examples["question"],
                                              examples["answers"]):
            text = answers["text"][0] if answers["text"] else ""
            starts = answers.get("answer_start") or []
            start = starts[0] if len(starts) else None
            ex = build_example(tokenizer, context, question, text, start, max_length)
            if ex is None:
                stats["dropped"] += 1
                continue
            out["input_ids"].append(ex["input_ids"])
            out["attention_mask"].append(ex["attention_mask"])
            out["labels"].append(ex["labels"])
            stats["kept"] += 1
        return out

    return preprocess_function, stats


def verify_alignment(tokenizer, dataset, n: int = 3, verbose: bool = True) -> None:
    """Assert the mask boundary is exact, and show it.

    This is the check behind the handout's "Label Misalignment" pitfall. It is
    cheap enough to run on every launch, and it fails loudly rather than letting
    a misaligned run produce believable curves.
    """
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        ids, labels = ex["input_ids"], ex["labels"]
        assert len(ids) == len(labels)
        supervised = [j for j, l in enumerate(labels) if l != -100]
        assert supervised, "example has no supervised tokens"
        assert supervised == list(range(supervised[0], len(labels))), \
            "supervised positions are not a contiguous suffix"
        assert all(labels[j] == ids[j] for j in supervised), "label/input mismatch"

        first = supervised[0]
        assert first > 0, "nothing is masked -- the prompt is being trained on"

        # Structural boundary checks. These catch an off-by-one in EITHER
        # direction, which the naive "is it whitespace?" test does not:
        #   * the last masked token must be the end of the scaffold ("Answer:"),
        #     so if the mask ran one token long it would end on the answer's
        #     first word instead;
        #   * the first supervised token must be a word start, which GPT-2 marks
        #     with a leading space, so if the mask stopped one token short it
        #     would land on the bare space or on ":".
        prev_tok = tokenizer.decode([ids[first - 1]])
        cur_tok = tokenizer.decode([ids[first]])
        assert prev_tok.rstrip().endswith(":"), (
            f"last masked token is {prev_tok!r}, expected the prompt scaffold to "
            "end at 'Answer:' -- the mask is one token too long and is eating the "
            "first token of the answer")
        assert cur_tok.startswith(" ") and cur_tok.strip(), (
            f"first supervised token is {cur_tok!r}, expected a space-prefixed "
            "word start -- the mask is one token too short")

        if verbose:
            tail = tokenizer.decode(ids[max(0, first - 4):first])
            print(f"    [align {i}] ...{tail!r} || {tokenizer.decode(ids[first:])!r}")


# =============================================================================
# 3. Task 2 -- Custom data collator (dynamic padding)
# =============================================================================

class QADataCollator:
    """[FIX-3] Dynamic padding to the batch maximum, as §4.2 requires.

    Padding every example to a fixed width during preprocessing is STATIC
    padding: a batch of short examples still costs a full-width forward pass.
    Padding to the batch maximum instead typically halves the tokens processed
    on SQuAD; `measure_padding_efficiency` computes the exact figure for the
    report's static-vs-dynamic analysis.

    `pad_to_multiple_of=8` keeps widths tensor-core friendly, costing a few
    tokens and returning more than that on fp16 hardware.

    Fills: `pad_token_id` for input_ids, 0 for attention_mask, -100 for labels --
    padded positions are therefore both unattended and unsupervised.
    """

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8,
                 max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict:
        width = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            width = ((width + m - 1) // m) * m
        if self.max_length:
            width = min(width, self.max_length)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])[:width]
            am = list(f["attention_mask"])[:width]
            lb = list(f["labels"])[:width]
            pad = width - len(ids)
            if pad > 0:
                ids += [self.pad_id] * pad
                am += [0] * pad
                lb += [-100] * pad
            input_ids.append(ids)
            attention_mask.append(am)
            labels.append(lb)

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def measure_padding_efficiency(dataset, batch_size: int, static_width: int) -> Dict:
    """Quantify dynamic vs static padding for the §4.2 analysis question."""
    lengths = [len(x) for x in dataset["input_ids"]]
    n_batches = int(np.ceil(len(lengths) / batch_size))
    dyn = sum(max(lengths[b * batch_size:(b + 1) * batch_size]) *
              len(lengths[b * batch_size:(b + 1) * batch_size])
              for b in range(n_batches))
    static = static_width * len(lengths)
    real = int(np.sum(lengths))
    return {"real_tokens": real,
            "dynamic_padded_tokens": int(dyn),
            "static_padded_tokens": int(static),
            "dynamic_waste_pct": round(100 * (1 - real / dyn), 2),
            "static_waste_pct": round(100 * (1 - real / static), 2),
            "token_reduction_vs_static": round(static / dyn, 2),
            "mean_len": round(float(np.mean(lengths)), 1),
            "p95_len": int(np.percentile(lengths, 95)),
            "max_len": int(np.max(lengths))}


# =============================================================================
# 4. Task 3 -- Model and LoRA construction
# =============================================================================

def load_tokenizer(model_name: str = "gpt2"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return tok


def load_base_model(model_name: str = "gpt2"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_name)


def build_lora_model(cfg: RunConfig, device):
    """Attach LoRA adapters to a frozen GPT-2.

    [FIX-10] `fan_in_fan_out=True`: GPT-2's `c_attn` / `c_proj` / `c_fc` are
    `transformers.pytorch_utils.Conv1D`, whose weight is stored (in, out) rather
    than `nn.Linear`'s (out, in). PEFT needs this to orient B and A correctly.
    Recent PEFT auto-detects Conv1D and overrides the flag with a warning;
    setting it explicitly is correct and silences the warning.

    PEFT zero-initialises B, so ΔW = 0 at step 0 and the adapted model starts
    exactly at the pre-trained function.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    set_global_seed(cfg.seed)
    model = load_base_model(cfg.model_name)
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        fan_in_fan_out=True,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.to(device)

    trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters -- LoRA did not attach"
    stray = [n for n in trainable if "lora" not in n]
    assert not stray, f"non-LoRA parameters are trainable: {stray[:3]}"
    return lora_model


def parameter_report(model) -> Dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "reduction_factor": round((total - trainable) / max(trainable, 1), 1)}


# =============================================================================
# 5. Training
# =============================================================================

def extract_curves(log_history: List[Dict]) -> Dict[str, List]:
    curves: Dict[str, List] = {"train_steps": [], "train_loss": [],
                               "eval_epochs": [], "eval_loss": []}
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            curves["train_steps"].append(entry.get("step", len(curves["train_steps"])))
            curves["train_loss"].append(entry["loss"])
        if "eval_loss" in entry:
            curves["eval_epochs"].append(entry.get("epoch", len(curves["eval_epochs"]) + 1))
            curves["eval_loss"].append(entry["eval_loss"])
    return curves


def _dir_size(path: str) -> int:
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, files in os.walk(path) for f in files)


def train_one(cfg: RunConfig, tokenizer, tok_train, tok_eval, output_dir: str,
              device, verbose: bool = True):
    """Train one configuration; return (model, record)."""
    set_global_seed(cfg.seed)
    model = build_lora_model(cfg, device)
    params = parameter_report(model)
    if verbose:
        print(f"  trainable {params['trainable_params']:,} / {params['total_params']:,} "
              f"({params['trainable_pct']}%), reduction {params['reduction_factor']}x")

    adapter_dir = os.path.join(output_dir, "model_checkpoints", cfg.tag())
    args = make_training_arguments(
        output_dir=os.path.join(output_dir, "_hf", cfg.tag()),
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        seed=cfg.seed,
        data_seed=cfg.seed,
        fp16=(device.type == "cuda"),
        dataloader_pin_memory=(device.type == "cuda"),
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=not verbose,
    )

    collator = QADataCollator(tokenizer, pad_to_multiple_of=8, max_length=cfg.max_length)
    trainer = make_trainer(model, args, tok_train, tok_eval, collator, tokenizer)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2 ** 20
               if device.type == "cuda" else None)

    eval_out = trainer.evaluate()
    model.save_pretrained(adapter_dir)

    eval_loss = float(eval_out.get("eval_loss", float("nan")))
    record = {
        "config": {**asdict(cfg), "target_modules": list(cfg.target_modules)},
        "params": params,
        "curves": extract_curves(trainer.state.log_history),
        "eval_loss": eval_loss,
        "perplexity": float(np.exp(min(eval_loss, 20.0))) if eval_loss == eval_loss else None,
        "cost": {"train_seconds": round(train_seconds, 1),
                 "seconds_per_epoch": round(train_seconds / max(cfg.num_train_epochs, 1), 1),
                 "peak_memory_mb": round(peak_mb, 1) if peak_mb else None},
        "adapter_path": adapter_dir,
        "adapter_size_kb": round(_dir_size(adapter_dir) / 1024, 1),
    }
    return model, record


# =============================================================================
# 6. Task 4a -- Generation
# =============================================================================

DECODING_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "greedy":      {"do_sample": False},
    "beam4":       {"do_sample": False, "num_beams": 4},
    "topk10":      {"do_sample": True, "top_k": 10, "temperature": 1.0},
    "topk25":      {"do_sample": True, "top_k": 25, "temperature": 1.0},
    "topk50":      {"do_sample": True, "top_k": 50, "temperature": 1.0},
    "nucleus0.8":  {"do_sample": True, "top_p": 0.80, "top_k": 0, "temperature": 1.0},
    "nucleus0.9":  {"do_sample": True, "top_p": 0.90, "top_k": 0, "temperature": 1.0},
    "nucleus0.95": {"do_sample": True, "top_p": 0.95, "top_k": 0, "temperature": 1.0},
    "temp0.7":     {"do_sample": True, "top_k": 50, "temperature": 0.7},
    "temp1.0":     {"do_sample": True, "top_k": 50, "temperature": 1.0},
    "temp1.3":     {"do_sample": True, "top_k": 50, "temperature": 1.3},
}


def clean_answer(text: str) -> str:
    """[FIX-6] Trim the continuation down to an answer span.

    GPT-2 does not reliably emit EOS, so generation runs to `max_new_tokens` and
    keeps going into a fabricated next Context/Question block. Without a stopping
    rule the F1 precision denominator explodes and scores collapse for reasons
    unrelated to answer quality -- the handout's "Generation Loops" pitfall.
    """
    text = text.split("\n")[0]
    for marker in ("Context:", "Question:", "Answer:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip().strip('"').strip()


def generate_answers(model, tokenizer, examples, device, max_length: int = 384,
                     max_new_tokens: int = 24, strategy: str = "greedy",
                     batch_size: int = 8, seed: Optional[int] = None) -> List[str]:
    """Batched generation.

    [FIX-7] The continuation is sliced off BY TOKEN COUNT, not by cutting the
    decoded string at `len(prompt)`. Character slicing breaks whenever the prompt
    was truncated or `skip_special_tokens` changed the decoded length, silently
    yielding empty or corrupted answers.

    [FIX-7b] Left padding: batched generation from a decoder-only model requires
    it, or the model continues from pad tokens rather than the real final token.
    `padding_side` is toggled and restored.

    [FIX-8] Inputs are placed on the model's device; the original ran the
    reloaded adapter on CPU.
    """
    if seed is not None:
        set_global_seed(seed)
    model.eval()
    gen_kwargs = dict(DECODING_STRATEGIES[strategy])
    gen_kwargs.update(max_new_tokens=max_new_tokens,
                      pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)

    prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    answers: List[str] = []
    try:
        with torch.no_grad():
            for i in range(0, len(examples), batch_size):
                chunk = examples[i:i + batch_size]
                prompts = [build_prompt_ids(tokenizer, ex["context"], ex["question"],
                                            budget=max_length - max_new_tokens)
                           for ex in chunk]
                width = max(len(p) for p in prompts)
                input_ids, attn = [], []
                for p in prompts:
                    pad = width - len(p)
                    input_ids.append([tokenizer.pad_token_id] * pad + p)
                    attn.append([0] * pad + [1] * len(p))
                batch = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                    "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
                }
                out = model.generate(**batch, **gen_kwargs)
                for row in out:
                    new_tokens = row[width:]           # token-count slice
                    answers.append(clean_answer(
                        tokenizer.decode(new_tokens, skip_special_tokens=True)))
    finally:
        tokenizer.padding_side = prev_side
    return answers


# =============================================================================
# 7. Task 4b -- SQuAD metrics, confidence intervals, significance testing
# =============================================================================

def normalize_answer(s: str) -> str:
    """Official SQuAD v1.1 normalisation: lowercase, drop punctuation, articles,
    and redundant whitespace."""
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def squad_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    if not pred or not gold:
        return float(pred == gold)
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def squad_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def per_example_scores(predictions: Sequence[str],
                       references: Sequence[Sequence[str]]):
    """SQuAD takes the MAX over the multiple gold answers per question."""
    em = np.array([max((squad_em(p, g) for g in golds), default=0.0)
                   for p, golds in zip(predictions, references)])
    f1 = np.array([max((squad_f1(p, g) for g in golds), default=0.0)
                   for p, golds in zip(predictions, references)])
    return em, f1


def bootstrap_ci(scores: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap over examples.

    §5.2 asks for confidence intervals. On a few hundred examples the interval is
    wide, and reporting that honestly is exactly the point -- it is what stops a
    two-point F1 difference being read as a result.
    """
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores), size=(n_boot, len(scores)))
    means = scores[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(scores.mean()), float(lo), float(hi)


def paired_bootstrap_test(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                          seed: int = 0) -> Dict[str, float]:
    """Two-sided paired bootstrap on the mean difference (a - b).

    Paired, because both systems are scored on the SAME examples; an unpaired
    test discards that structure and is far less powerful.
    """
    diff = np.asarray(a) - np.asarray(b)
    if len(diff) == 0:
        return {"mean_diff": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    centred = boot - boot.mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": observed,
            "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(np.mean(np.abs(centred) >= abs(observed)))}


def _distinct_n(texts: Sequence[str], n: int = 1) -> float:
    """Diversity measure for the §4.4 decoding comparison."""
    grams, total = set(), 0
    for t in texts:
        toks = normalize_answer(t).split()
        for i in range(max(0, len(toks) - n + 1)):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return round(len(grams) / max(total, 1), 4)


def evaluate_model(model, tokenizer, test_ds, device, cfg: RunConfig,
                   strategy: Optional[str] = None, n_boot: int = 2000,
                   verbose: bool = True) -> Dict[str, Any]:
    strategy = strategy or cfg.decoding
    examples = [test_ds[i] for i in range(len(test_ds))]
    t0 = time.time()
    preds = generate_answers(model, tokenizer, examples, device,
                             max_length=cfg.max_length,
                             max_new_tokens=cfg.max_new_tokens,
                             strategy=strategy,
                             batch_size=cfg.eval_batch_size,
                             seed=cfg.seed)
    gen_seconds = time.time() - t0
    refs = [ex["answers"]["text"] for ex in examples]
    em, f1 = per_example_scores(preds, refs)
    em_mean, em_lo, em_hi = bootstrap_ci(em, n_boot=n_boot, seed=cfg.seed)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1, n_boot=n_boot, seed=cfg.seed)
    result = {
        "strategy": strategy,
        "n": len(preds),
        "exact_match": round(100 * em_mean, 2),
        "em_ci95": [round(100 * em_lo, 2), round(100 * em_hi, 2)],
        "f1": round(100 * f1_mean, 2),
        "f1_ci95": [round(100 * f1_lo, 2), round(100 * f1_hi, 2)],
        "mean_pred_words": round(float(np.mean([len(p.split()) for p in preds])), 2),
        "empty_pred_rate": round(float(np.mean([not p.strip() for p in preds])), 4),
        "distinct_1": _distinct_n(preds, 1),
        "generation_seconds": round(gen_seconds, 1),
        "_per_example": {"em": em.tolist(), "f1": f1.tolist()},
        "_predictions": preds,
    }
    if verbose:
        print(f"  [{strategy:>11}] EM {result['exact_match']:5.2f} {result['em_ci95']}"
              f"   F1 {result['f1']:5.2f} {result['f1_ci95']}")
    return result


def cross_check_with_hf_metric(preds, examples) -> Optional[Dict]:
    """Optional agreement check against `evaluate`'s SQuAD implementation.

    The local implementation IS the official algorithm, but cross-checking costs
    nothing when the library is present, and the pipeline must not die when it is
    absent (offline cluster, no network).
    """
    try:
        import evaluate as hf_evaluate
        metric = hf_evaluate.load("squad")
        return metric.compute(
            predictions=[{"id": ex["id"], "prediction_text": p}
                         for p, ex in zip(preds, examples)],
            references=[{"id": ex["id"], "answers": ex["answers"]} for ex in examples])
    except Exception as exc:                                   # pragma: no cover
        print(f"  (hf `evaluate` cross-check unavailable: {type(exc).__name__})")
        return None


# =============================================================================
# 8. Failure analysis
# =============================================================================

def failure_analysis(examples, predictions, f1_scores, out_path: str,
                     n_show: int = 15) -> Dict[str, int]:
    """Categorise and dump the worst predictions.

    Categories separate *format* failures (the model never learned to answer)
    from *comprehension* failures (it answered, wrongly) -- the distinction that
    matters when interpreting a 124M model's scores.
    """
    cats: Counter = Counter()
    rows = []
    for ex, pred, f1 in zip(examples, predictions, f1_scores):
        gold = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        norm_pred = normalize_answer(pred)
        gold_len = max(len(normalize_answer(gold).split()), 1)
        if not norm_pred:
            cat = "empty"
        elif len(norm_pred.split()) > 3 * gold_len:
            cat = "verbose / ran on"
        elif norm_pred not in normalize_answer(ex["context"]):
            cat = "not extractive (hallucinated span)"
        elif f1 == 0:
            cat = "wrong span"
        elif f1 < 1.0:
            cat = "partial overlap"
        else:
            cat = "correct"
        cats[cat] += 1
        rows.append((f1, ex["question"], gold, pred, cat))

    rows.sort(key=lambda r: r[0])
    total = max(sum(cats.values()), 1)
    lines = ["# Failure analysis", "", "## Category counts", ""]
    lines += [f"- **{c}**: {n} ({100 * n / total:.1f}%)" for c, n in cats.most_common()]
    lines += ["", f"## {n_show} lowest-F1 predictions", ""]
    for f1, q, gold, pred, cat in rows[:n_show]:
        lines += [f"**Q:** {q}", f"- gold: `{gold}`", f"- pred: `{pred}`",
                  f"- F1 {f1:.2f} — _{cat}_", ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    return dict(cats)


# =============================================================================
# 9. Plots
# =============================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(records: Dict[str, Dict], out_path: str, title: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, rec in records.items():
        c = rec["curves"]
        if c["train_loss"]:
            axes[0].plot(c["train_steps"], c["train_loss"], lw=1.1, label=name)
        if c["eval_loss"]:
            axes[1].plot(c["eval_epochs"], c["eval_loss"], marker="o", ms=4, label=name)
    axes[0].set_title("Training loss"); axes[0].set_xlabel("step")
    axes[1].set_title("Validation loss"); axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.set_ylabel("cross-entropy"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ablation(rows: List[Dict], x_key: str, out_path: str, xlabel: str,
                  logx: bool = False) -> None:
    if not rows:
        return
    plt = _plt()
    rows = sorted(rows, key=lambda r: r[x_key])
    x = [r[x_key] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    lo = [r["f1"] - r["f1_ci95"][0] for r in rows]
    hi = [r["f1_ci95"][1] - r["f1"] for r in rows]
    axes[0].errorbar(x, [r["f1"] for r in rows], yerr=[lo, hi], marker="o", capsize=4)
    axes[0].set_ylabel("SQuAD F1 (95% CI)"); axes[0].set_title("Quality")
    axes[1].plot(x, [r["trainable_params"] for r in rows], marker="s", color="#C44E52")
    axes[1].set_ylabel("trainable parameters"); axes[1].set_yscale("log")
    axes[1].set_title("Capacity")
    axes[2].plot(x, [r["train_seconds"] for r in rows], marker="^", color="#55A868")
    axes[2].set_ylabel("training seconds"); axes[2].set_title("Cost")
    for ax in axes:
        ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_params_vs_performance(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        ax.errorbar(r["trainable_params"], r["f1"],
                    yerr=[[r["f1"] - r["f1_ci95"][0]], [r["f1_ci95"][1] - r["f1"]]],
                    marker="o", capsize=3)
        ax.annotate(r["name"], (r["trainable_params"], r["f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("SQuAD F1 (95% CI)")
    ax.set_title("Parameter count vs. performance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_decoding_study(results: List[Dict], out_path: str) -> None:
    if not results:
        return
    plt = _plt()
    names = [r["strategy"] for r in results]
    f1 = [r["f1"] for r in results]
    err = [[r["f1"] - r["f1_ci95"][0] for r in results],
           [r["f1_ci95"][1] - r["f1"] for r in results]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(names))
    axes[0].bar(xs, f1, yerr=err, capsize=3, color="#4C72B0")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("SQuAD F1 (95% CI)")
    axes[0].set_title("Answer quality by decoding strategy")
    axes[1].scatter([r["distinct_1"] for r in results], f1)
    for r in results:
        axes[1].annotate(r["strategy"], (r["distinct_1"], r["f1"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)
    axes[1].set_xlabel("distinct-1 (diversity)"); axes[1].set_ylabel("F1")
    axes[1].set_title("Quality vs. diversity")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# 10. Orchestration
# =============================================================================

def _free(model, device) -> None:
    del model
    gc.collect()
    if _HAS_TORCH and device.type == "cuda":
        torch.cuda.empty_cache()


def prepare_data(cfg: RunConfig, tokenizer, verbose: bool = True):
    train_raw, eval_raw, test_raw = load_squad_splits(cfg)
    fn, stats = make_preprocess_fn(tokenizer, cfg.max_length)
    tok_train = train_raw.map(fn, batched=True, batch_size=64,
                              remove_columns=train_raw.column_names,
                              desc="tokenising train")
    tok_eval = eval_raw.map(fn, batched=True, batch_size=64,
                            remove_columns=eval_raw.column_names,
                            desc="tokenising eval")
    if verbose:
        seen = max(stats["kept"] + stats["dropped"], 1)
        print(f"  train {len(tok_train)}  eval {len(tok_eval)}  test {len(test_raw)}")
        print(f"  dropped {stats['dropped']} ({100 * stats['dropped'] / seen:.2f}%) "
              f"— answer could not fit within {cfg.max_length} tokens")
        print("  label-mask alignment check:")
        verify_alignment(tokenizer, tok_train, n=3)
    return tok_train, tok_eval, test_raw, stats


def run_single(cfg: RunConfig, tokenizer, tok_train, tok_eval, test_raw, device,
               out_dir: str, evaluate_now: bool = True):
    print(f"\n=== run: {cfg.tag()} ===")
    model, record = train_one(cfg, tokenizer, tok_train, tok_eval, out_dir, device)
    if evaluate_now:
        record["metrics"] = evaluate_model(model, tokenizer, test_raw, device, cfg)
    return model, record


def run_baseline_comparison(cfg: RunConfig, tokenizer, test_raw, device,
                            ft_metrics: Dict) -> Dict:
    """§4.4 -- compare against the non-fine-tuned model.

    Without it the fine-tuned scores are uninterpretable: GPT-2 emits something
    for any prompt, and some of it overlaps the gold answer by chance. The paired
    bootstrap says whether the improvement is real.
    """
    print("\n=== baseline: non-fine-tuned GPT-2 (zero-shot) ===")
    base = load_base_model(cfg.model_name).to(device)
    base_metrics = evaluate_model(base, tokenizer, test_raw, device, cfg)
    test = {
        "f1": paired_bootstrap_test(np.array(ft_metrics["_per_example"]["f1"]),
                                    np.array(base_metrics["_per_example"]["f1"]),
                                    seed=cfg.seed),
        "em": paired_bootstrap_test(np.array(ft_metrics["_per_example"]["em"]),
                                    np.array(base_metrics["_per_example"]["em"]),
                                    seed=cfg.seed),
    }
    print(f"  fine-tuned − baseline F1: {100 * test['f1']['mean_diff']:+.2f} "
          f"(p = {test['f1']['p_value']:.4f})")
    _free(base, device)
    return {"baseline_metrics": base_metrics, "significance": test}


def run_ablations(base_cfg: RunConfig, tokenizer, tok_train, tok_eval, test_raw,
                  device, out_dir: str) -> Dict[str, Dict]:
    """§4.3 -- vary one axis at a time from a fixed baseline.

    One-factor-at-a-time rather than the full 4x3x2 grid: 24 runs is not
    affordable on Colab, and at a single seed the cross-product could not support
    interaction claims anyway. State this in the methodology section -- it is
    sound design, not a shortcut.

    `lora_alpha` is scaled with `r` so the effective scaling α/r stays constant
    across the rank sweep; otherwise the rank ablation silently varies the update
    magnitude too, and the comparison is confounded.
    """
    records: Dict[str, Dict] = {}
    seen_tags = set()

    configs: List[RunConfig] = []
    for r in (4, 8, 16, 32):
        configs.append(base_cfg.replace(name=f"rank{r}", lora_r=r, lora_alpha=2 * r))
    for lr in (1e-4, 2e-4, 5e-4):
        configs.append(base_cfg.replace(name=f"lr{lr:g}", learning_rate=lr))
    for preset, mods in TARGET_PRESETS.items():
        configs.append(base_cfg.replace(name=f"target_{preset}", target_modules=mods))

    for cfg in configs:
        if cfg.tag() in seen_tags:            # identical to an earlier run
            continue
        seen_tags.add(cfg.tag())
        model, rec = run_single(cfg, tokenizer, tok_train, tok_eval, test_raw,
                                device, out_dir)
        records[cfg.name] = rec
        _free(model, device)
    return records


def run_decoding_study(model, tokenizer, test_raw, device, cfg: RunConfig) -> List[Dict]:
    """§4.4 -- greedy / beam / top-k / nucleus / temperature on one model."""
    print("\n=== decoding strategy study ===")
    return [evaluate_model(model, tokenizer, test_raw, device, cfg,
                           strategy=s, n_boot=1000)
            for s in DECODING_STRATEGIES]


def strip_heavy(obj: Any) -> Any:
    """Drop per-example arrays before writing the summary JSON."""
    if isinstance(obj, dict):
        return {k: strip_heavy(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_heavy(v) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA fine-tuning of GPT-2 on SQuAD")
    p.add_argument("--mode", default="single",
                   choices=["selftest", "smoke", "single", "ablation", "decoding", "all"])
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--model-name", default="gpt2")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--targets", default="attn", choices=list(TARGET_PRESETS))
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "selftest":
        run_selftest()
        return

    if args.mode == "smoke":
        args.epochs, args.n_train, args.n_eval, args.n_test = 1.0, 128, 32, 32
        args.max_length, args.batch_size = 256, 4
        args.output_dir = os.path.join(args.output_dir, "_smoke")

    cfg = RunConfig(
        name="baseline", model_name=args.model_name, max_length=args.max_length,
        n_train=args.n_train, n_eval=args.n_eval, n_test=args.n_test,
        lora_r=args.rank, lora_alpha=2 * args.rank,
        target_modules=TARGET_PRESETS[args.targets],
        learning_rate=args.lr, num_train_epochs=args.epochs,
        train_batch_size=args.batch_size, eval_batch_size=args.batch_size,
        seed=args.seed,
    )
    for sub in ("model_checkpoints", "plots", "runs"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    set_global_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected — this will be very slow.")

    tokenizer = load_tokenizer(cfg.model_name)
    tok_train, tok_eval, test_raw, drop_stats = prepare_data(cfg, tokenizer)

    padding = measure_padding_efficiency(tok_train, cfg.train_batch_size, cfg.max_length)
    print(f"  padding: dynamic wastes {padding['dynamic_waste_pct']}% of tokens vs "
          f"{padding['static_waste_pct']}% static "
          f"({padding['token_reduction_vs_static']}x fewer tokens processed)")

    summary: Dict[str, Any] = {
        "config": {**asdict(cfg), "target_modules": list(cfg.target_modules)},
        "data": {"dropped": drop_stats, "padding": padding},
        "runs": {},
    }

    model, record = run_single(cfg, tokenizer, tok_train, tok_eval, test_raw,
                               device, args.output_dir)
    summary["runs"]["baseline"] = record
    summary["baseline_comparison"] = run_baseline_comparison(
        cfg, tokenizer, test_raw, device, record["metrics"])

    examples = [test_raw[i] for i in range(len(test_raw))]
    summary["failure_categories"] = failure_analysis(
        examples, record["metrics"]["_predictions"],
        record["metrics"]["_per_example"]["f1"],
        os.path.join(args.output_dir, "failure_analysis.md"))
    hf_check = cross_check_with_hf_metric(record["metrics"]["_predictions"], examples)
    if hf_check:
        summary["hf_metric_cross_check"] = hf_check

    if args.mode in ("decoding", "all"):
        dec = run_decoding_study(model, tokenizer, test_raw, device, cfg)
        summary["decoding_study"] = dec
        plot_decoding_study(dec, os.path.join(args.output_dir, "plots", "decoding.png"))
    _free(model, device)

    if args.mode in ("ablation", "all"):
        abl = run_ablations(cfg, tokenizer, tok_train, tok_eval, test_raw,
                            device, args.output_dir)
        summary["runs"].update(abl)
        plot_training_curves(abl, os.path.join(args.output_dir, "plots", "curves.png"),
                             "Ablation training dynamics")
        rank_rows = [{"name": k, "lora_r": v["config"]["lora_r"],
                      "f1": v["metrics"]["f1"], "f1_ci95": v["metrics"]["f1_ci95"],
                      "trainable_params": v["params"]["trainable_params"],
                      "train_seconds": v["cost"]["train_seconds"]}
                     for k, v in abl.items() if k.startswith("rank")]
        lr_rows = [{"name": k, "learning_rate": v["config"]["learning_rate"],
                    "f1": v["metrics"]["f1"], "f1_ci95": v["metrics"]["f1_ci95"],
                    "trainable_params": v["params"]["trainable_params"],
                    "train_seconds": v["cost"]["train_seconds"]}
                   for k, v in abl.items() if k.startswith("lr")]
        plots = os.path.join(args.output_dir, "plots")
        plot_ablation(rank_rows, "lora_r", os.path.join(plots, "ablation_rank.png"),
                      "LoRA rank r")
        plot_ablation(lr_rows, "learning_rate", os.path.join(plots, "ablation_lr.png"),
                      "learning rate", logx=True)
        plot_params_vs_performance(rank_rows + lr_rows,
                                   os.path.join(plots, "params_vs_performance.png"))

    for name, rec in summary["runs"].items():
        with open(os.path.join(args.output_dir, "runs", f"{name}.json"), "w") as fh:
            json.dump(strip_heavy(rec), fh, indent=2)
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as fh:
        json.dump(strip_heavy(summary), fh, indent=2)

    print("\n================ SUMMARY ================")
    for name, rec in summary["runs"].items():
        m = rec.get("metrics", {})
        print(f"{name:>16}  F1 {m.get('f1', float('nan')):5.2f} {m.get('f1_ci95')}  "
              f"EM {m.get('exact_match', float('nan')):5.2f}  "
              f"params {rec['params']['trainable_params']:>9,}  "
              f"{rec['cost']['train_seconds']:.0f}s")
    print(f"\nartefacts written to {os.path.abspath(args.output_dir)}")


# =============================================================================
# 11. Offline self-test (no model download, no GPU)
# =============================================================================

class _MockTokenizer:
    """Whitespace tokenizer that reproduces GPT-2's leading-space merge, so the
    boundary logic is exercised under exactly the condition that breaks the naive
    implementation."""

    eos_token_id = 999
    pad_token_id = 999
    padding_side = "right"

    def __init__(self):
        self.vocab: Dict[str, int] = {"<eos>": 999}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab))

    def __call__(self, text, add_special_tokens=False):
        toks, buf = [], ""
        for ch in text:
            if ch == " ":
                if buf:
                    toks.append(buf)
                buf = " "
            elif ch == "\n":
                if buf:
                    toks.append(buf)
                buf = ""
                toks.append("\n")
            else:
                buf += ch
        if buf:
            toks.append(buf)
        return {"input_ids": [self._id(t) for t in toks]}

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        out = []
        for i in ids:
            tok = inv.get(int(i), "")
            if tok == "<eos>":
                tok = "" if skip_special_tokens else "<eos>"
            out.append(tok)
        return "".join(out)


def run_selftest() -> None:
    """Exercise every component that does not need network access."""
    print("== preprocessing / label alignment ==")
    tok = _MockTokenizer()
    ctx = "Rome is the capital of Italy. " * 40
    ex = build_example(tok, ctx, "What is the capital of Italy?", "Rome", 0, 64)
    assert ex is not None
    n_sup = sum(1 for l in ex["labels"] if l != -100)
    assert n_sup == ex["n_answer"], (n_sup, ex["n_answer"])
    first = next(i for i, l in enumerate(ex["labels"]) if l != -100)
    assert tok.decode([ex["input_ids"][first]]).strip() == "Rome", \
        "first supervised token is not the answer's first token"
    assert len(ex["input_ids"]) <= 64
    assert all(ex["labels"][j] == ex["input_ids"][j]
               for j in range(first, len(ex["labels"])))
    print(f"   boundary exact: {n_sup} supervised tokens, first = "
          f"{tok.decode([ex['input_ids'][first]])!r}")

    long_ctx = "filler " * 500 + "The answer is Zanzibar."
    ex2 = build_example(tok, long_ctx, "Where?", "Zanzibar",
                        answer_start=len(long_ctx) - 12, max_length=64)
    assert ex2 is not None
    assert any(tok.decode([i]).strip() == "Zanzibar" for i in ex2["input_ids"]), \
        "answer-centred window did not retain the answer"
    print("   long-context window keeps the answer inside the prompt")

    assert build_example(tok, ctx, "q", "", None, 64) is None
    assert build_example(tok, ctx, "q", "x " * 200, None, 64) is None
    print("   empty and over-long answers dropped rather than silently truncated")

    # verify_alignment must reject a mask that is off by one in EITHER direction.
    class _OneRow(list):
        def __getitem__(self, i):
            return list.__getitem__(self, i)

    verify_alignment(tok, _OneRow([ex]), n=1, verbose=False)
    for shift, why in ((+1, "one token too long"), (-1, "one token too short")):
        broken = dict(ex)
        labels = list(ex["labels"])
        boundary = next(i for i, l in enumerate(labels) if l != -100)
        if shift > 0:
            labels[boundary] = -100
        else:
            labels[boundary - 1] = ex["input_ids"][boundary - 1]
        broken["labels"] = labels
        try:
            verify_alignment(tok, _OneRow([broken]), n=1, verbose=False)
            raise SystemExit(f"verify_alignment failed to catch a mask {why}")
        except AssertionError:
            pass
    print("   verify_alignment rejects off-by-one masks in both directions")

    print("== collator ==")
    if _HAS_TORCH:
        feats = [build_example(tok, "short ctx", "q?", "yes", 0, 64),
                 build_example(tok, ctx, "q?", "Rome", 0, 64)]
        batch = QADataCollator(tok, pad_to_multiple_of=8)(feats)
        assert batch["input_ids"].shape == batch["labels"].shape
        assert batch["input_ids"].shape[1] % 8 == 0
        pad_pos = batch["attention_mask"] == 0
        assert bool((batch["labels"][pad_pos] == -100).all()), "padding leaked into labels"
        assert bool((batch["input_ids"][pad_pos] == tok.pad_token_id).all())
        widths = {len(f["input_ids"]) for f in feats}
        assert batch["input_ids"].shape[1] < 64 or max(widths) > 56, \
            "width should track the batch, not a fixed maximum"
        print(f"   dynamic width {tuple(batch['input_ids'].shape)}; "
              "padding masked out of the loss")
    else:
        print("   (torch unavailable — collator test skipped)")

    print("== metrics ==")
    assert squad_em("The Rome.", "rome") == 1.0
    assert abs(squad_f1("Rome Italy", "Rome") - 2 / 3) < 1e-9
    assert squad_f1("", "Rome") == 0.0
    em, f1 = per_example_scores(["Rome", "Paris"], [["Rome", "roma"], ["London"]])
    assert list(em) == [1.0, 0.0] and f1[1] == 0.0
    print("   EM/F1 match the official normalisation; max-over-golds works")

    scores = np.array([1.0] * 50 + [0.0] * 50)
    mean, lo, hi = bootstrap_ci(scores, n_boot=2000, seed=0)
    assert abs(mean - 0.5) < 1e-9 and lo < 0.5 < hi
    print(f"   bootstrap CI {mean:.2f} [{lo:.2f}, {hi:.2f}]")

    rng = np.random.default_rng(0)
    a = rng.random(200)
    real = paired_bootstrap_test(a, a - 0.2, seed=0)
    null = paired_bootstrap_test(a, a.copy(), seed=0)
    assert real["p_value"] < 0.05 and abs(real["mean_diff"] - 0.2) < 0.02
    assert null["p_value"] > 0.5
    print(f"   paired bootstrap: real diff p={real['p_value']:.4f}, "
          f"null diff p={null['p_value']:.2f}")

    print("== generation cleanup ==")
    assert clean_answer(" Rome\nContext: something else") == "Rome"
    assert clean_answer(" Rome Question: next one") == "Rome"
    assert clean_answer("   ") == ""
    print("   stopping rule trims runaway continuations")

    print("== padding efficiency ==")
    fake = {"input_ids": [[0] * n for n in [10, 12, 200, 14, 16, 11, 13, 15]]}
    stats = measure_padding_efficiency(fake, batch_size=4, static_width=384)
    assert stats["static_waste_pct"] > stats["dynamic_waste_pct"]
    print(f"   dynamic {stats['dynamic_waste_pct']}% waste vs "
          f"static {stats['static_waste_pct']}% "
          f"({stats['token_reduction_vs_static']}x)")

    print("== config plumbing ==")
    c = RunConfig()
    c2 = c.replace(lora_r=32, lora_alpha=64)
    assert c.lora_r == 8 and c2.lora_r == 32 and c.tag() != c2.tag()
    print(f"   {c.tag()} -> {c2.tag()}")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
