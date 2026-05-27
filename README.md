# MotionLM v5 — Multi-Agent Motion Forecasting

> A Transformer-based model for predicting future trajectories of autonomous vehicles and surrounding agents, featuring interleaved temporal–social encoding, rotary position embeddings, and soft winner-takes-all training.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#2-training)
  - [3. Resuming Training](#3-resuming-training)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Loss Functions & Metrics](#loss-functions--metrics)
- [Checkpoints & Logging](#checkpoints--logging)
- [Key Design Decisions](#key-design-decisions)
- [Citation](#citation)

---

## Overview

MotionLM v5 forecasts **K = 6 future trajectory modes** for the ego vehicle (AV) over a **3-second horizon** (T_future = 30 steps), conditioned on **2-second history** (T_hist = 20 steps) for up to **3 agents** (AV + scored agent + nearest neighbor).

Key highlights:

- **InterleavedEncoder** — alternates per-agent temporal self-attention (RoPE) with cross-agent social attention at every layer
- **RelativeGeometryEncoder** — attention-weighted spatial bias using pairwise distance and bearing (#S5)
- **Gated social residual** — per-layer learnable scalar gate `α` controls social influence (#S6)
- **Non-autoregressive ParallelDecoder** — outputs all K modes and timesteps in a single forward pass
- **EMA weights** — updated every optimizer step (#E11), used for validation
- **Soft-WTA + diversity loss** — encourages mode diversity while penalizing the closest prediction (#L9, #L10)

---

## Architecture

<img width="654" height="787" alt="image" src="https://github.com/user-attachments/assets/489c4485-1f07-48ab-aad3-e80835d6f1d8" />


---

## Project Structure

<img width="900" height="590" alt="image" src="https://github.com/user-attachments/assets/f02b58e4-4108-4e5c-b8de-2421481ada36" />


---

## Requirements

```bash
pip install torch torchvision torchaudio   # PyTorch ≥ 2.1 recommended
pip install numpy tqdm
```

> **Note:** `torch.compile` and `torch.nn.functional.scaled_dot_product_attention` (FlashAttention path) require PyTorch ≥ 2.0.

Tested on:
- Python 3.10+
- PyTorch 2.2 (CUDA 12.1)
- Google Colab (T4 / A100)

---

## Quick Start

### 1. Preprocessing

Convert the raw Argoverse-format `.tar.gz` archive into a fast-loading float16 numpy cache:

```bash
python preprocess.py \
  --input  /path/to/train_data.tar.gz \
  --output /path/to/cache/train

python preprocess.py \
  --input  /path/to/val_data.tar.gz \
  --output /path/to/cache/val
```

This produces two files per split:

| File | Shape | dtype |
|------|-------|-------|
| `hist.npy` | `(N_scenes, 3, 20, 7)` | float16 |
| `fut.npy` | `(N_scenes, 3, 30, 2)` | float16 |

Add `--force` to rebuild an existing cache.

**Feature layout (F=7):**

| Index | Feature |
|-------|---------|
| 0 | x (ego-normalized) |
| 1 | y (ego-normalized) |
| 2 | vx |
| 3 | vy |
| 4 | speed |
| 5 | heading (radians) |
| 6 | type_id (0=AV, 1=Agent, 2=Other) |

> Heading is expanded to `(cos_h, sin_h)` at model input time by `HeadingExpander`, giving F_proj = 8.

---

### 2. Training

```bash
python train.py \
  --train_cache /path/to/cache/train \
  --val_cache   /path/to/cache/val \
  --ckpt_dir    /path/to/checkpoints \
  --log_csv     /path/to/train_log.csv \
  --epochs      30 \
  --batch       256 \
  --lr          3e-4
```

Optional flags:

```bash
  --compile          # Enable torch.compile (PyTorch 2.x, faster on modern GPUs)
  --sanity_check     # Validate NaN/Inf in cache before training starts
  --workers N        # DataLoader workers (default: min(8, cpu_count))
```

Sample training output:

```
  Ep  | Loss    | trADE   trFDE   trMR    tr1ADE  | vaADE   vaFDE   vaMR    va1ADE  va3ADE  | Skip  Time
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
     1 |  1.2341 |   1.823   3.941   0.512   2.105 |   1.654   3.612   0.481   1.892   1.701 |    0  142.3s  ← best
     2 |  1.0872 |   1.612   3.502   0.473   1.934 |   1.521   3.341   0.452   1.743   1.582 |    0  139.7s  ← best
```

---

### 3. Resuming Training

```bash
python train.py \
  --resume /path/to/checkpoints/motionlm_v5_last.pt \
  [... other args ...]
```

---

## Configuration

All hyperparameters are defined as constants at the top of `train.py` and can also be passed as CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_HIST` | 20 | History timesteps |
| `T_FUTURE` | 30 | Future timesteps to predict |
| `N_MAX` | 3 | Max agents per scene |
| `K` | 6 | Number of trajectory modes |
| `D_MODEL` | 128 | Transformer hidden dimension |
| `N_HEADS` | 4 | Attention heads |
| `N_LAYERS` | 4 | Encoder layers |
| `D_FF` | 256 | Feed-forward dimension |
| `DROPOUT` | 0.1 | Dropout rate |
| `SOFT_WTA_TEMP` | 1.0 | Temperature for soft-WTA loss |
| `DIVERSITY_COEFF` | 0.02 | Weight of diversity loss |
| `CLS_COEFF` | 0.5 | Weight of classification loss term |
| `EMA_DECAY` | 0.999 | EMA decay for model weights |
| `--lr` | 3e-4 | Peak learning rate |
| `--warmup_steps` | 500 | Linear warmup steps |
| `--epochs` | 30 | Total training epochs |
| `--batch` | 256 | Batch size |

---

## Model Details

### InterleavedEncoder

Each of the 4 encoder layers runs:

1. **TemporalBlock** — causal-free self-attention over T_hist timesteps per agent, with Rotary Position Embeddings (RoPE) applied to Q and K.
2. **RelativeGeometryEncoder** — encodes pairwise (distance, cos_angle, sin_angle) between agents; aggregates using softmax attention weighted by `exp(-dist / scale)` so nearby agents contribute more (#S5).
3. **SocialBlock** — standard multi-head cross-agent attention with key-padding mask for padded agents.
4. **Gated broadcast** — `x += α.abs() * social_summary` where `α` is a per-layer learnable scalar initialized to 0.1 (#S6).

### ParallelDecoder

- Learns `K` mode embeddings and `T_future` time embeddings.
- Queries are formed as `mode_emb[k] + time_emb[t]`, reshaped to `(K × T_future, D)`.
- Two TransformerDecoder layers cross-attend to the encoder memory.
- Separate `conf_head` produces per-mode confidence logits for classification.

### EMA

Exponential moving average of model weights (decay = 0.999) is updated **every optimizer step** (not every epoch). EMA weights are applied during validation and saved alongside the live model in checkpoints.

---

## Loss Functions & Metrics

### Training Loss

```
L_total = L_soft-WTA + L_diversity
```

**Soft Winner-Takes-All (L_soft-WTA):**
- Computes per-mode ADE against ground truth.
- Soft assignment weights via `softmax(-ADE / temperature)`.
- Regression term: weighted sum of per-mode ADEs.
- Classification term: cross-entropy toward the hard-argmin winner mode.

**Diversity Loss:**
- Penalizes modes whose final endpoints are within `margin = 2.0 m` of each other.
- Encourages the decoder to spread predictions across the plausible future space.

### Validation Metrics

| Metric | Description |
|--------|-------------|
| `minADE@K` | Min ADE over all K modes |
| `minFDE@K` | Min FDE over all K modes |
| `MissRate` | Fraction with best-mode FDE > 2.0 m |
| `top1ADE` | ADE of the highest-confidence mode |
| `top3ADE` | Min ADE among top-3 confidence modes |

---

## Checkpoints & Logging

**Checkpoints** are saved after every epoch to `ckpt_dir/`:

- `motionlm_v5_last.pt` — always the latest epoch
- `motionlm_v5_best.pt` — epoch with the lowest `val minADE`

Each checkpoint contains: `epoch`, `step`, `model` state dict, `ema` shadow weights, `optim` state dict, and all validation metrics.

**CSV log** (`train_log.csv`) records per-epoch: `epoch`, `lr`, `tr_loss`, `tr_ade`, `tr_fde`, `tr_mr`, `tr_t1`, `val_ade`, `val_fde`, `val_mr`, `val_t1`, `val_t3`, `nan_skip`, `time_s`.

---

## Key Design Decisions

| Tag | Description |
|-----|-------------|
| `#S5` | Attention-weighted geometry aggregation (distance-softmax instead of mean) |
| `#S6` | Per-layer gated social residual with learnable scalar α |
| `#E11` | EMA updated per optimizer step, not per epoch |
| `#L9` | Diversity coefficient = 0.02 |
| `#L10` | Soft-WTA temperature = 1.0 |
| `#V12` | Confidence-weighted ADE metric (top-1 and top-3 modes) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{motionlmv5,
  title   = {MotionLM v5: Interleaved Temporal-Social Transformer for Multi-Agent Motion Forecasting},
  year    = {2025},
  url     = {https://github.com/your-username/motionlm-v5}
}
```

---

*Built with PyTorch. Designed for Google Colab (T4/A100) with Google Drive cache storage.*
