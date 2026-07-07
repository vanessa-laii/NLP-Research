"""
Bottleneck Adapter Hyperparameter Sweep with Annotator Embedding Matrix
========================================================================
Same grid as bottleneck_sweep.py (m ∈ {16,32,64,128,256}) but each sample
is also given the rater_id of the human annotator who labeled it.

Architecture
------------
  text_rep    = ModernBERT CLS token          [batch, 768]
  ann_rep     = Embedding(rater_id)           [batch, 32]
  combined    = concat(text_rep, ann_rep)     [batch, 800]
  logits      = Linear(800, 6)                [batch, 6]

The annotator embedding matrix (82 × 32) is trained jointly with the
bottleneck adapter and the classifier head.

Fixed config (best from hyperparameter sweep)
---------------------------------------------
  m (bottleneck dim) : 256

Total runs: 1

Usage
-----
  python bottleneck_annotator_sweep.py
  python bottleneck_annotator_sweep.py --data_dir ../../dataset --output_dir ./bn_ann_runs

  # Limit to a specific GPU:
  CUDA_VISIBLE_DEVICES=0 python bottleneck_annotator_sweep.py
"""

import argparse
import csv
import gc
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import adapters
import numpy as np
import pandas as pd
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from adapters import SeqBnConfig
from sklearn.metrics import f1_score, hamming_loss, multilabel_confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

torch._dynamo.config.suppress_errors = True


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "answerdotai/ModernBERT-base"
HIDDEN_SIZE    = 768
ANNOTATOR_DIM  = 32
MAX_LENGTH     = 128
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 2e-5
THRESHOLD      = 0.5
SEED           = 42

EKMAN_EMOTIONS = ["joy", "anger", "sadness", "fear", "disgust", "surprise"]

M_VALUES = [16, 32, 64, 128, 256]   


# ── DATASET ───────────────────────────────────────────────────────────────────

class EmotionDatasetWithAnnotator(Dataset):
    """EmotionDataset that also carries a remapped annotator index."""

    def __init__(self, encodings, labels, annotator_ids):
        self.encodings      = encodings
        self.labels         = torch.tensor(labels, dtype=torch.float32)
        self.annotator_ids  = torch.tensor(annotator_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"]       = self.labels[idx]
        item["annotator_id"] = self.annotator_ids[idx]
        return item


def load_data(data_dir: str):
    """Load splits, filter to Ekman rows, and build a rater_id → index map."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    missing = [e for e in EKMAN_EMOTIONS if e not in train_df.columns]
    if missing:
        raise ValueError(f"Ekman emotion columns missing from CSV: {missing}")

    emotion_cols = EKMAN_EMOTIONS

    def keep_ekman_rows(df):
        mask = df[emotion_cols].any(axis=1)
        return df[mask].reset_index(drop=True)

    train_df = keep_ekman_rows(train_df)
    val_df   = keep_ekman_rows(val_df)
    test_df  = keep_ekman_rows(test_df)

    # Build a stable rater_id → contiguous index mapping from ALL splits combined
    all_rater_ids  = sorted(set(train_df["rater_id"]) | set(val_df["rater_id"]) | set(test_df["rater_id"]))
    rater_to_idx   = {rid: i for i, rid in enumerate(all_rater_ids)}
    num_annotators = len(rater_to_idx)

    num_labels = len(emotion_cols)

    print(f"Loaded  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(f"Labels ({num_labels}): {emotion_cols}")
    print(f"Annotators: {num_annotators}  (rater_id mapped to 0–{num_annotators-1})")

    return train_df, val_df, test_df, emotion_cols, num_labels, rater_to_idx, num_annotators


def build_datasets(train_df, val_df, test_df, emotion_cols, rater_to_idx, tokenizer):
    """Tokenise all splits and wrap them in EmotionDatasetWithAnnotator."""
    def tokenize(texts):
        return tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH, padding=False)

    train_enc = tokenize(train_df["text"])
    val_enc   = tokenize(val_df["text"])
    test_enc  = tokenize(test_df["text"])

    def labels(df):
        return df[emotion_cols].values.astype(np.float32)

    def ann_ids(df):
        return df["rater_id"].map(rater_to_idx).values.astype(np.int64)

    return (
        EmotionDatasetWithAnnotator(train_enc, labels(train_df), ann_ids(train_df)),
        EmotionDatasetWithAnnotator(val_enc,   labels(val_df),   ann_ids(val_df)),
        EmotionDatasetWithAnnotator(test_enc,  labels(test_df),  ann_ids(test_df)),
    )


@dataclass
class AnnotatorDataCollator:
    """Pads tokenizer fields and stacks labels + annotator_id."""
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        annotator_ids = torch.stack([f.pop("annotator_id") for f in features])
        labels        = torch.stack([f.pop("labels") for f in features])

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["annotator_id"] = annotator_ids
        batch["labels"]       = labels
        return batch


# ── MODEL ─────────────────────────────────────────────────────────────────────

class AnnotatorAwareBnModel(nn.Module):
    """ModernBERT backbone with Bottleneck Adapter + annotator embedding matrix.

    Forward:
        text_rep  = CLS token from backbone            [batch, 768]
        ann_rep   = embedding lookup for rater_id      [batch, 32]
        combined  = concat(text_rep, ann_rep)          [batch, 800]
        logits    = classifier(combined)               [batch, 6]
    """

    def __init__(self, backbone, num_annotators: int, annotator_dim: int, num_labels: int):
        super().__init__()
        self.backbone            = backbone
        self.annotator_embedding = nn.Embedding(num_annotators, annotator_dim)
        self.classifier          = nn.Linear(HIDDEN_SIZE + annotator_dim, num_labels)
        self.num_labels          = num_labels

    def forward(self, input_ids, attention_mask, annotator_id, labels=None, **kwargs):
        out      = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_rep = out.last_hidden_state[:, 0]                    # CLS token
        ann_rep  = self.annotator_embedding(annotator_id)
        combined = torch.cat([text_rep, ann_rep], dim=-1)
        logits   = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))


def build_model(num_labels: int, num_annotators: int, m: int, device: str):
    """Fresh ModernBERT + Bottleneck Adapter + annotator embedding for one m."""
    backbone = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")

    adapters.init(backbone)
    reduction_factor = HIDDEN_SIZE / m
    bn_cfg = SeqBnConfig(reduction_factor=reduction_factor)
    backbone.add_adapter("bn_adapter", config=bn_cfg)
    backbone.train_adapter("bn_adapter")
    backbone.set_active_adapters("bn_adapter")

    # Bypass ModernBERT's torch.compile MLP path — incompatible with adapters
    for module in backbone.modules():
        if hasattr(module, "compiled_mlp") and hasattr(module, "mlp_norm_and_mlp"):
            module.compiled_mlp = module.mlp_norm_and_mlp

    model = AnnotatorAwareBnModel(backbone, num_annotators, ANNOTATOR_DIM, num_labels)

    for param in model.annotator_embedding.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  m={m}  reduction_factor={reduction_factor:.1f}")
    print(f"  Params total={total:.1f}M  trainable={trainable:.3f}M "
          f"({trainable / total * 100:.2f}%)")
    print(f"  Annotator embedding table: {num_annotators} × {ANNOTATOR_DIM} = "
          f"{num_annotators * ANNOTATOR_DIM:,} params")

    # ── Debug: verify which layers are unfrozen ───────────────────────────────
    print("\n  [DEBUG] Unfrozen parameter groups:")
    frozen_count    = 0
    unfrozen_count  = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    TRAINABLE  {name:60s}  shape={list(param.shape)}")
            unfrozen_count += 1
        else:
            frozen_count += 1
    print(f"\n  [DEBUG] Summary: {unfrozen_count} trainable groups, "
          f"{frozen_count} frozen groups")

    emb_trainable = all(p.requires_grad for p in model.annotator_embedding.parameters())
    print(f"  [DEBUG] annotator_embedding unfrozen: {emb_trainable}")
    clf_trainable = all(p.requires_grad for p in model.classifier.parameters())
    print(f"  [DEBUG] classifier unfrozen:          {clf_trainable}\n")

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


# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, title_suffix: str, save_path: str):
    """Plot a 2×3 grid of per-label binary confusion matrices and save as PNG.

    For multi-label classification each emotion is treated as an independent
    binary classification problem, so each cell shows:
        TN  FP
        FN  TP
    """
    mcm = multilabel_confusion_matrix(labels, preds)   # shape: (6, 2, 2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Per-Label Confusion Matrices  —  {title_suffix}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for i, (ax, emotion) in enumerate(zip(axes.flat, EKMAN_EMOTIONS)):
        cm = mcm[i]                                    # 2×2: [[TN, FP],[FN, TP]]
        total = cm.sum()

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(emotion.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted 0", "Predicted 1"], fontsize=9)
        ax.set_yticklabels(["Actual 0", "Actual 1"], fontsize=9)

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        thresh = cm.max() / 2.0
        for row in range(2):
            for col in range(2):
                pct = cm[row, col] / total * 100 if total > 0 else 0.0
                ax.text(
                    col, row,
                    f"{cell_labels[row][col]}\n{cm[row, col]:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[row, col] > thresh else "black",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


# ── LOSS CURVE ────────────────────────────────────────────────────────────────

def extract_loss_history(log_history: list):
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

    epochs       = sorted(val_by_epoch.keys())
    train_losses = [train_sums[ep] / train_counts[ep] for ep in epochs if ep in train_sums]
    val_losses   = [val_by_epoch[ep] for ep in epochs]
    return train_losses, val_losses, epochs


def plot_loss_curve(train_losses, val_losses, epochs, m, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    if train_losses:
        ax.plot(epochs[:len(train_losses)], train_losses,
                label="Train loss", color="#2563EB", linewidth=2, marker="o", markersize=4)

    ax.plot(epochs, val_losses,
            label="Validation loss", color="#F97316", linewidth=2,
            linestyle="--", marker="s", markersize=4)

    best_idx = int(np.argmin(val_losses))
    ax.axvline(epochs[best_idx], color="#F97316", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.annotate(
        f" best val\n epoch {epochs[best_idx]}\n {val_losses[best_idx]:.4f}",
        xy=(epochs[best_idx], val_losses[best_idx]),
        xytext=(epochs[best_idx] + 0.3, val_losses[best_idx] * 1.04),
        fontsize=8, color="#F97316",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title(
        f"Training & Validation Loss  —  Bottleneck Adapter + Annotator Embedding  m={m}",
        fontsize=13, fontweight="bold",
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
    num_annotators: int,
    tokenizer,
    output_dir: str,
    device: str,
    epochs: int,
) -> tuple:
    run_dir = os.path.join(output_dir, f"m{m}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN  bottleneck m={m}  + annotator embedding")
    print(f"{'='*60}")

    model = build_model(num_labels, num_annotators, m, device)

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
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = AnnotatorDataCollator(tokenizer),
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  Training finished in {elapsed/60:.1f} min")

    train_losses, val_losses, ep_axis = extract_loss_history(trainer.state.log_history)
    best_val_loss = float(min(val_losses)) if val_losses else float("nan")
    best_epoch    = ep_axis[int(np.argmin(val_losses))] if val_losses else -1

    train_by_epoch = {ep: train_losses[i] for i, ep in enumerate(ep_axis) if i < len(train_losses)}
    epoch_rows = []
    for ep, vl in zip(ep_axis, val_losses):
        epoch_rows.append({
            "m": m, "epoch": ep,
            "train_loss": round(train_by_epoch.get(ep, float("nan")), 6),
            "val_loss":   round(vl, 6),
        })

    curve_path = os.path.join(run_dir, f"loss_curve_m{m}.png")
    if val_losses:
        plot_loss_curve(train_losses, val_losses, ep_axis, m, curve_path)
    else:
        print("  No validation loss recorded — skipping loss curve.")

    test_preds_output = trainer.predict(test_dataset)
    logits = test_preds_output.predictions
    labels = test_preds_output.label_ids
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (probs >= THRESHOLD).astype(int)

    micro_f1 = float(f1_score(labels, preds, average="micro",  zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average="macro",  zero_division=0))
    h_loss   = float(hamming_loss(labels, preds))

    print(f"  TEST  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}  hamming_loss={h_loss:.4f}")

    cm_path = os.path.join(run_dir, f"confusion_matrix_m{m}.png")
    plot_confusion_matrix(labels, preds, f"Bottleneck Adapter + Annotator  m={m}", cm_path)

    del model, trainer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "m":                 m,
        "reduction_factor":  round(HIDDEN_SIZE / m, 2),
        "annotator_dim":     ANNOTATOR_DIM,
        "test_micro_f1":     micro_f1,
        "test_macro_f1":     macro_f1,
        "test_hamming_loss": h_loss,
        "best_val_loss":     best_val_loss,
        "best_epoch":        best_epoch,
        "train_minutes":     round(elapsed / 60, 2),
    }
    return summary, epoch_rows


# ── RESULTS LOGGING ───────────────────────────────────────────────────────────

def append_result(results_path, row, write_header):
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  sweep_results.csv updated → {results_path}")


def append_epoch_losses(epoch_losses_path, rows, write_header):
    if not rows:
        return
    with open(epoch_losses_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  epoch_losses.csv updated  → {epoch_losses_path}")


def write_report(report_path, all_results, epoch_losses_path):
    sep  = "=" * 70
    dash = "-" * 70
    lines = [
        sep,
        "  Bottleneck Adapter + Annotator Embedding Sweep — Results Report",
        "  Model : answerdotai/ModernBERT-base",
        f"  Annotator embedding dim: {ANNOTATOR_DIM}  (82 raters × {ANNOTATOR_DIM})",
        "  Task  : Multi-label Ekman emotion classification (6 classes)",
        "  Labels: joy, anger, sadness, fear, disgust, surprise",
        sep, "",
        "── TEST SET METRICS (sorted by micro F1) ──────────────────────────────", "",
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
        f"  (full data in {os.path.basename(epoch_losses_path)})", "",
        f"  {'m':>5}  {'epoch':>6}  {'train_loss':>12}  {'val_loss':>10}",
        dash,
    ]
    if os.path.exists(epoch_losses_path):
        with open(epoch_losses_path, newline="") as f:
            for entry in csv.DictReader(f):
                lines.append(
                    f"  {entry['m']:>5}  {entry['epoch']:>6}  "
                    f"{entry['train_loss']:>12}  {entry['val_loss']:>10}"
                )
    lines += ["", sep]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  sweep_report.txt written  → {report_path}")


def write_final_results(output_dir: str, all_results: list):
    """Write a concise final_results.txt with the key numbers for easy download."""
    path = os.path.join(output_dir, "final_results.txt")
    sep  = "=" * 60
    best = max(all_results, key=lambda x: x["test_micro_f1"])

    lines = [
        sep,
        "  Bottleneck Adapter + Annotator Embedding — Final Results",
        f"  Model         : {MODEL_NAME}",
        f"  Annotator dim : {ANNOTATOR_DIM}  (embedding table: 81 × {ANNOTATOR_DIM})",
        f"  Loss function : Binary Cross-Entropy with Logits (BCE)",
        f"  Threshold     : {THRESHOLD}",
        sep,
        "",
        "  ALL RUNS (sorted by micro F1):",
        f"  {'m':>5}  {'red_factor':>11}  {'micro_F1':>10}  {'macro_F1':>10}  "
        f"{'hamming':>10}  {'best_val_loss':>14}  {'best_epoch':>11}  {'mins':>6}",
        "-" * 95,
    ]

    for row in sorted(all_results, key=lambda x: x["test_micro_f1"], reverse=True):
        lines.append(
            f"  {row['m']:>5}  {row['reduction_factor']:>11.1f}  "
            f"{row['test_micro_f1']:>10.4f}  {row['test_macro_f1']:>10.4f}  "
            f"{row['test_hamming_loss']:>10.4f}  "
            f"{row['best_val_loss']:>14.4f}  {row['best_epoch']:>11}  "
            f"{row['train_minutes']:>6.1f}"
        )

    lines += [
        "",
        sep,
        "  BEST CONFIG:",
        f"    m                  = {best['m']}",
        f"    reduction_factor   = {best['reduction_factor']}",
        f"    annotator_dim      = {best['annotator_dim']}",
        f"    test_micro_F1      = {best['test_micro_f1']:.4f}",
        f"    test_macro_F1      = {best['test_macro_f1']:.4f}",
        f"    test_hamming_loss  = {best['test_hamming_loss']:.4f}",
        f"    best_val_loss      = {best['best_val_loss']:.4f}",
        f"    best_epoch         = {best['best_epoch']}",
        f"    train_minutes      = {best['train_minutes']}",
        sep,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  final_results.txt written → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Bottleneck + annotator embedding sweep")
    parser.add_argument("--data_dir",   default=os.path.join(here, "..", "..", "dataset"))
    parser.add_argument("--output_dir", default=os.path.join(here, "bn_ann_runs"))
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
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

    train_df, val_df, test_df, emotion_cols, num_labels, rater_to_idx, num_annotators = \
        load_data(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, val_dataset, test_dataset = build_datasets(
        train_df, val_df, test_df, emotion_cols, rater_to_idx, tokenizer
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
            train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
            num_labels=num_labels, num_annotators=num_annotators,
            tokenizer=tokenizer, output_dir=args.output_dir,
            device=device, epochs=args.epochs,
        )
        all_results.append(summary)
        append_result(results_path, summary, write_header=(i == 0))
        append_epoch_losses(epoch_losses_path, epoch_rows, write_header=(i == 0))

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
    write_final_results(args.output_dir, all_results)

    print(f"\nOutput files in {args.output_dir}:")
    print(f"  sweep_results.csv    — one row per run")
    print(f"  epoch_losses.csv     — per-epoch train/val loss")
    print(f"  sweep_report.txt     — human-readable summary")
    print(f"  final_results.txt    — concise final results for easy download")
    print(f"  m*/loss_curve_*.png  — loss curve per run")


if __name__ == "__main__":
    main()
