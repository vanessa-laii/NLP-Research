"""
Bottleneck Adapter Hyperparameter Sweep — ModernBERT on GoEmotions (Ekman emotions)
=====================================================================================
Sweeps bottleneck dimension (m) for multi-label emotion classification restricted
to the 6 Ekman emotions: joy, anger, sadness, fear, disgust, surprise

Formula:  h <- W_up · f(W_down · h) + h
  W_down : d × m   (projects hidden size d down to bottleneck m)
  W_up   : m × d   (projects back up)
  f      : GeLU non-linearity
  h      : residual skip connection

Hyperparameter grid
-------------------
  m (bottleneck dim) : [16, 32, 64, 128, 256]

Total runs: 5

Usage
-----
  python bottleneck_sweep.py
  python bottleneck_sweep.py --data_dir ../dataset --output_dir ./bn_runs --epochs 10

  # Limit to a specific GPU (e.g. GPU 0):
  CUDA_VISIBLE_DEVICES=0 python bottleneck_sweep.py
"""

import argparse
import csv
import gc
import math
import os
import time
from collections import defaultdict

# Keep HuggingFace Trainer from auto-wrapping the model in DataParallel when
# the server exposes multiple GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import adapters
import numpy as np
import torch
import torch._dynamo
from adapters import SeqBnConfig
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ModernBERT uses torch.compile for its MLP path. With adapter layers active,
# that compiled path can fail inside torch._dynamo; fall back to eager.
torch._dynamo.config.suppress_errors = True

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required: pip install pandas")


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "answerdotai/ModernBERT-base"
HIDDEN_SIZE  = 768          # ModernBERT-base hidden dimension
MAX_LENGTH   = 128
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 2e-5
THRESHOLD    = 0.5
SEED         = 42

# The 6 Ekman emotions present in the GoEmotions dataset
EKMAN_EMOTIONS = ["joy", "anger", "sadness", "fear", "disgust", "surprise"]

# Hyperparameter grid: bottleneck dimensions to sweep
M_VALUES = [16, 32, 64, 128, 256]


# ── DATASET ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_data(data_dir: str):
    """Load train / validation / test splits from local CSV files.

    Only the 6 Ekman emotion columns are kept as labels.
    Rows where none of the 6 Ekman emotions are active are dropped.
    """
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    missing = [e for e in EKMAN_EMOTIONS if e not in train_df.columns]
    if missing:
        raise ValueError(f"Ekman emotion columns missing from CSV: {missing}")

    emotion_cols = EKMAN_EMOTIONS

    def keep_ekman_rows(df: pd.DataFrame) -> pd.DataFrame:
        mask = df[emotion_cols].any(axis=1)
        return df[mask].reset_index(drop=True)

    train_df = keep_ekman_rows(train_df)
    val_df   = keep_ekman_rows(val_df)
    test_df  = keep_ekman_rows(test_df)

    num_labels = len(emotion_cols)

    print(f"Loaded  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}  "
          f"(rows with ≥1 Ekman label)")
    print(f"Labels ({num_labels}): {emotion_cols}")

    return train_df, val_df, test_df, emotion_cols, num_labels


def build_datasets(train_df, val_df, test_df, emotion_cols, tokenizer):
    """Tokenise all splits and wrap them in EmotionDataset."""
    def tokenize(texts):
        return tokenizer(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_enc = tokenize(train_df["text"])
    val_enc   = tokenize(val_df["text"])
    test_enc  = tokenize(test_df["text"])

    def labels(df):
        return df[emotion_cols].values.astype(np.float32)

    return (
        EmotionDataset(train_enc, labels(train_df)),
        EmotionDataset(val_enc,   labels(val_df)),
        EmotionDataset(test_enc,  labels(test_df)),
    )


# ── MODEL ─────────────────────────────────────────────────────────────────────

def build_model(num_labels: int, m: int, device: str):
    """Fresh ModernBERT + bottleneck adapter for a given bottleneck dim m.

    SeqBnConfig uses reduction_factor = hidden_size / m, which sets the
    bottleneck to exactly m units.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    adapters.init(model)

    reduction_factor = HIDDEN_SIZE / m
    bn_cfg = SeqBnConfig(reduction_factor=reduction_factor)
    model.add_adapter("bn_adapter", config=bn_cfg)
    model.train_adapter("bn_adapter")
    model.set_active_adapters("bn_adapter")

    # Use ModernBERT's eager MLP path instead of compiled_mlp to avoid a
    # torch._dynamo crash when adapter layers are active.
    for module in model.modules():
        if hasattr(module, "compiled_mlp") and hasattr(module, "mlp_norm_and_mlp"):
            module.compiled_mlp = module.mlp_norm_and_mlp

    # Unfreeze classification head (randomly initialised)
    for name, param in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            param.requires_grad = True

    model.to(device)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  m={m}  reduction_factor={reduction_factor:.1f}")
    print(f"  Params total={total:.1f}M  trainable={trainable:.2f}M "
          f"({trainable / total * 100:.1f}%)")

    return model


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "micro_f1":     f1_score(labels, preds, average="micro",  zero_division=0),
        "macro_f1":     f1_score(labels, preds, average="macro",  zero_division=0),
        "hamming_loss": hamming_loss(labels, preds),
    }


# ── LOSS CURVE ────────────────────────────────────────────────────────────────

def extract_loss_history(log_history: list):
    """Parse Trainer log_history into per-epoch train and val loss lists."""
    train_sums   = defaultdict(float)
    train_counts = defaultdict(int)
    val_by_epoch = {}

    for entry in log_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        ep_int = math.ceil(ep) if ep % 1 > 0 else int(ep)

        if "eval_loss" in entry:
            val_by_epoch[ep_int] = entry["eval_loss"]
        elif "loss" in entry:
            train_sums[ep_int]   += entry["loss"]
            train_counts[ep_int] += 1

    epochs = sorted(val_by_epoch.keys())
    train_losses = [
        train_sums[ep] / train_counts[ep]
        for ep in epochs
        if ep in train_sums
    ]
    val_losses = [val_by_epoch[ep] for ep in epochs]
    return train_losses, val_losses, epochs


def plot_loss_curve(
    train_losses: list,
    val_losses: list,
    epochs: list,
    m: int,
    save_path: str,
):
    """Save a training/validation loss curve PNG for bottleneck dim m."""
    fig, ax = plt.subplots(figsize=(9, 5))

    if train_losses:
        ax.plot(
            epochs[: len(train_losses)],
            train_losses,
            label="Train loss",
            color="#2563EB",
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.plot(
        epochs,
        val_losses,
        label="Validation loss",
        color="#F97316",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
    )

    best_idx = int(np.argmin(val_losses))
    ax.axvline(epochs[best_idx], color="#F97316", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.annotate(
        f" best val\n epoch {epochs[best_idx]}\n {val_losses[best_idx]:.4f}",
        xy=(epochs[best_idx], val_losses[best_idx]),
        xytext=(epochs[best_idx] + 0.3, val_losses[best_idx] * 1.04),
        fontsize=8,
        color="#F97316",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(
        f"Training & Validation Loss  —  Bottleneck Adapter  m={m}",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(left=epochs[0])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved → {save_path}")


# ── SINGLE RUN ────────────────────────────────────────────────────────────────

def train_one_config(
    m: int,
    train_dataset,
    val_dataset,
    test_dataset,
    num_labels: int,
    tokenizer,
    output_dir: str,
    device: str,
    epochs: int,
) -> tuple:
    """Train one bottleneck dim m and return (summary dict, epoch_rows list)."""
    run_dir = os.path.join(output_dir, f"m{m}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN  bottleneck m={m}")
    print(f"{'='*60}")

    model = build_model(num_labels, m, device)

    training_args = TrainingArguments(
        output_dir                  = run_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "micro_f1",
        greater_is_better           = True,
        logging_steps               = 100,
        fp16                        = (device == "cuda"),
        seed                        = SEED,
        report_to                   = "none",
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = tokenizer,
        data_collator    = DataCollatorWithPadding(tokenizer),
        compute_metrics  = compute_metrics,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  Training finished in {elapsed/60:.1f} min")

    # ── Extract loss history ──────────────────────────────────────────────────
    train_losses, val_losses, ep_axis = extract_loss_history(
        trainer.state.log_history
    )
    best_val_loss = float(min(val_losses)) if val_losses else float("nan")
    best_epoch    = ep_axis[int(np.argmin(val_losses))] if val_losses else -1

    # Build per-epoch rows for epoch_losses.csv
    train_by_epoch = {
        ep: train_losses[i]
        for i, ep in enumerate(ep_axis)
        if i < len(train_losses)
    }
    epoch_rows = []
    for ep, vl in zip(ep_axis, val_losses):
        epoch_rows.append({
            "m":          m,
            "epoch":      ep,
            "train_loss": round(train_by_epoch.get(ep, float("nan")), 6),
            "val_loss":   round(vl, 6),
        })

    # ── Loss curve ────────────────────────────────────────────────────────────
    curve_path = os.path.join(run_dir, f"loss_curve_m{m}.png")
    if val_losses:
        plot_loss_curve(train_losses, val_losses, ep_axis, m, curve_path)
    else:
        print("  No validation loss recorded — skipping loss curve.")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    test_preds_output = trainer.predict(test_dataset)
    logits = test_preds_output.predictions
    labels = test_preds_output.label_ids
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (probs >= THRESHOLD).astype(int)

    micro_f1 = float(f1_score(labels, preds, average="micro",  zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average="macro",  zero_division=0))
    h_loss   = float(hamming_loss(labels, preds))

    print(f"  TEST  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}  "
          f"hamming_loss={h_loss:.4f}")

    # ── Free GPU memory before next run ──────────────────────────────────────
    del model, trainer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "m":                 m,
        "reduction_factor":  round(HIDDEN_SIZE / m, 2),
        "test_micro_f1":     micro_f1,
        "test_macro_f1":     macro_f1,
        "test_hamming_loss": h_loss,
        "best_val_loss":     best_val_loss,
        "best_epoch":        best_epoch,
        "train_minutes":     round(elapsed / 60, 2),
    }
    return summary, epoch_rows


# ── RESULTS LOGGING ───────────────────────────────────────────────────────────

def append_result(results_path: str, row: dict, write_header: bool):
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  sweep_results.csv updated → {results_path}")


def append_epoch_losses(epoch_losses_path: str, rows: list, write_header: bool):
    if not rows:
        return
    with open(epoch_losses_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  epoch_losses.csv updated  → {epoch_losses_path}")


def write_report(report_path: str, all_results: list, epoch_losses_path: str):
    """Write a human-readable sweep_report.txt consolidating all metrics."""
    sep  = "=" * 70
    dash = "-" * 70

    lines = [
        sep,
        "  Bottleneck Adapter Hyperparameter Sweep — Results Report",
        "  Model : answerdotai/ModernBERT-base",
        "  Task  : Multi-label Ekman emotion classification (6 classes)",
        "  Labels: joy, anger, sadness, fear, disgust, surprise",
        sep,
        "",
        "── TEST SET METRICS (sorted by micro F1) ──────────────────────────────",
        "",
        f"  {'m':>5}  {'red_factor':>11}  {'micro_F1':>10}  {'macro_F1':>10}  "
        f"{'hamming':>10}  {'best_val_loss':>14}  {'best_epoch':>11}  {'mins':>6}",
        dash,
    ]

    for row in sorted(all_results, key=lambda x: x["test_micro_f1"], reverse=True):
        lines.append(
            f"  {row['m']:>5}  {row['reduction_factor']:>11.1f}  "
            f"{row['test_micro_f1']:>10.4f}  {row['test_macro_f1']:>10.4f}  "
            f"{row['test_hamming_loss']:>10.4f}  "
            f"{row['best_val_loss']:>14.4f}  {row['best_epoch']:>11}  "
            f"{row['train_minutes']:>6.1f}"
        )

    best = max(all_results, key=lambda x: x["test_micro_f1"])
    lines += [
        "",
        f"  Best config → m={best['m']}  (micro_F1={best['test_micro_f1']:.4f})",
        "",
        "── PER-EPOCH LOSSES ───────────────────────────────────────────────────",
        f"  (full data in {os.path.basename(epoch_losses_path)})",
        "",
        f"  {'m':>5}  {'epoch':>6}  {'train_loss':>12}  {'val_loss':>10}",
        dash,
    ]

    if os.path.exists(epoch_losses_path):
        with open(epoch_losses_path, newline="") as f:
            reader = csv.DictReader(f)
            for entry in reader:
                lines.append(
                    f"  {entry['m']:>5}  {entry['epoch']:>6}  "
                    f"{entry['train_loss']:>12}  {entry['val_loss']:>10}"
                )

    lines += ["", sep]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  sweep_report.txt written  → {report_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Bottleneck adapter hyperparameter sweep")
    parser.add_argument(
        "--data_dir",
        default=os.path.join(here, "..", "dataset"),
        help="Directory containing train.csv, validation.csv, test.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(here, "bn_runs"),
        help="Root directory for per-run checkpoints and outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Max training epochs per run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_df, val_df, test_df, emotion_cols, num_labels = load_data(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, val_dataset, test_dataset = build_datasets(
        train_df, val_df, test_df, emotion_cols, tokenizer
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results_path      = os.path.join(args.output_dir, "sweep_results.csv")
    epoch_losses_path = os.path.join(args.output_dir, "epoch_losses.csv")
    report_path       = os.path.join(args.output_dir, "sweep_report.txt")

    for p in (results_path, epoch_losses_path, report_path):
        if os.path.exists(p):
            os.remove(p)

    print(f"\nBottleneck dims to sweep: {M_VALUES}")
    print(f"Total runs: {len(M_VALUES)}\n")

    all_results = []
    for i, m in enumerate(M_VALUES):
        summary, epoch_rows = train_one_config(
            m=m,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_labels=num_labels,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
        )
        all_results.append(summary)
        append_result(results_path, summary, write_header=(i == 0))
        append_epoch_losses(epoch_losses_path, epoch_rows, write_header=(i == 0))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SWEEP COMPLETE — Results summary")
    print(f"{'='*60}")
    print(f"{'m':>5}  {'red_factor':>11}  {'micro_f1':>10}  {'macro_f1':>10}  {'hamming':>10}")
    print("-" * 60)
    for row in sorted(all_results, key=lambda x: x["test_micro_f1"], reverse=True):
        print(f"{row['m']:>5}  {row['reduction_factor']:>11.1f}  "
              f"{row['test_micro_f1']:>10.4f}  {row['test_macro_f1']:>10.4f}  "
              f"{row['test_hamming_loss']:>10.4f}")

    best = max(all_results, key=lambda x: x["test_micro_f1"])
    print(f"\nBest config → m={best['m']}  (micro_f1={best['test_micro_f1']:.4f})")

    write_report(report_path, all_results, epoch_losses_path)

    print(f"\nOutput files in {args.output_dir}:")
    print(f"  sweep_results.csv    — one row per run (test F1 + val loss)")
    print(f"  epoch_losses.csv     — train/val loss for every epoch of every run")
    print(f"  sweep_report.txt     — human-readable summary of everything above")
    print(f"  m*/loss_curve_*.png  — loss curve plot per run")


if __name__ == "__main__":
    main()
