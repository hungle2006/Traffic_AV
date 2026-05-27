from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
T_HIST   = 20
T_FUTURE = 30
N_MAX    = 3
F_DIM    = 7   # default = preprocessing gốc; auto-detect khi load
K        = 6

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 256
DROPOUT  = 0.1

SOFT_WTA_TEMP   = 1.0
DIVERSITY_COEFF = 0.02
CLS_COEFF       = 0.5
EMA_DECAY       = 0.999

TRAIN_CACHE = "/content/drive/MyDrive/train"
VAL_CACHE   = "/content/drive/MyDrive/val"
CKPT_DIR    = "/content/drive/MyDrive/checkpoints"
LOG_CSV     = "/content/drive/MyDrive/train_log.csv"


# ═══════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════
class CachedMotionDataset(Dataset):
    """
    Đọc cache từ preprocessing gốc.

    hist.npy  : (N, 3, T_HIST, F)   — float16
    fut.npy   : (N, 3, T_FUTURE, 2) — float16
    n_agents.npy : tuỳ chọn; nếu không có → N=3 cho tất cả scenes

    Preprocessing đã ego-normalize → KHÔNG normalize lại ở đây.
    Cast float16 → float32 trong __getitem__.
    Agent 0 = AV (agent cần predict).
    """
    def __init__(self, cache_dir: str, sanity_check: bool = False):
        cache_dir = Path(cache_dir)
        h_path = cache_dir / "hist.npy"
        f_path = cache_dir / "fut.npy"
        n_path = cache_dir / "n_agents.npy"

        if not (h_path.exists() and f_path.exists()):
            raise FileNotFoundError(
                f"Cache không tồn tại tại '{cache_dir}'.\n"
                "Chạy preprocessing trước để tạo hist.npy / fut.npy."
            )

        self.hist     = np.load(h_path, mmap_mode="r")
        self.fut      = np.load(f_path, mmap_mode="r")
        self.n_agents = (
            np.load(n_path)
            if n_path.exists()
            else np.full(len(self.hist), N_MAX, dtype=np.int32)
        )

        # Detect thực tế từ cache
        self._n_scenes = len(self.hist)
        self._f_dim    = self.hist.shape[3]
        self._dtype    = self.hist.dtype

        assert self.hist.shape[2] == T_HIST,   \
            f"T_HIST mismatch: got {self.hist.shape[2]}, expect {T_HIST}"
        assert self.fut.shape[2]  == T_FUTURE, \
            f"T_FUTURE mismatch: got {self.fut.shape[2]}, expect {T_FUTURE}"

        print(f"  [Dataset] {self._n_scenes:>8,} scenes"
              f"  hist{tuple(self.hist.shape[1:])} ({self._dtype})"
              f"  fut{tuple(self.fut.shape[1:])}"
              f"  | {cache_dir.name}")

        if sanity_check:
            self._sanity_check()

    def _sanity_check(self):
        print("  [Sanity] Checking NaN/Inf ...", end=" ", flush=True)
        h = np.array(self.hist, dtype=np.float32)
        f = np.array(self.fut,  dtype=np.float32)
        issues = []
        if np.isnan(h).any(): issues.append("hist-NaN")
        if np.isinf(h).any(): issues.append("hist-Inf")
        if np.isnan(f).any(): issues.append("fut-NaN")
        if np.isinf(f).any(): issues.append("fut-Inf")
        if issues:
            print(f"WARNING: {', '.join(issues)}")
        else:
            print(f"OK  hist[{h.min():.2f}, {h.max():.2f}]"
                  f"  fut[{f.min():.2f}, {f.max():.2f}]")

    def __len__(self):
        return self._n_scenes

    def __getitem__(self, idx):
        n = int(self.n_agents[idx])
        # Cast float16 → float32 (training cần float32)
        h = self.hist[idx, :n].astype(np.float32)   # (n, T_HIST, F)
        f = self.fut [idx, :n].astype(np.float32)   # (n, T_FUTURE, 2)
        return torch.from_numpy(h), torch.from_numpy(f), n


def collate_fn(batch):
    """
    Pad về N_MAX agents.
    mask[b, i] = True  → padding (ignore trong attention)
    mask[b, i] = False → real agent
    """
    hists, futs, ns = zip(*batch)
    B     = len(ns)
    f_dim = hists[0].shape[-1]

    hist_pad = torch.zeros(B, N_MAX, T_HIST,   f_dim)
    fut_pad  = torch.zeros(B, N_MAX, T_FUTURE, 2)
    mask     = torch.ones (B, N_MAX, dtype=torch.bool)

    for i, (h, f, n) in enumerate(zip(hists, futs, ns)):
        hist_pad[i, :n] = h
        fut_pad [i, :n] = f
        mask    [i, :n] = False

    return hist_pad, fut_pad, mask, torch.tensor(ns, dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════════
# HEADING EXPANDER
# ═══════════════════════════════════════════════════════════════════════════
class HeadingExpander(nn.Module):
    """
    Cache gốc F_DIM=7: channel 5 = scalar heading (radian).
    Expand sang (cos_h, sin_h) → output F_DIM=8.
    Layout in : [x, y, vx, vy, speed, heading, type_id]
    Layout out: [x, y, vx, vy, speed, cos_h, sin_h, type_id]
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h     = x[..., 5:6]                          # scalar heading
        cos_h = torch.cos(h)
        sin_h = torch.sin(h)
        return torch.cat([x[..., :5], cos_h, sin_h, x[..., 6:]], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# ROTARY POSITION EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        assert dim % 2 == 0
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t     = torch.arange(max_len).float()
        freqs = torch.outer(t, theta)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_emb", emb.cos()[None, None])
        self.register_buffer("sin_emb", emb.sin()[None, None])

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1] // 2
        return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

    def forward(self, q, k, seq_len):
        cos = self.cos_emb[:, :, :seq_len]
        sin = self.sin_emb[:, :, :seq_len]
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)


# ═══════════════════════════════════════════════════════════════════════════
# RELATIVE GEOMETRY ENCODER  (#S5: attention-weighted)
# ═══════════════════════════════════════════════════════════════════════════
class RelativeGeometryEncoder(nn.Module):
    """
    Encode khoảng cách + góc tương đối giữa các agent.
    #S5: Aggregation theo softmax(-dist) thay mean — gần = ảnh hưởng nhiều hơn.
    Padding agents bị zero-out ở output.
    """
    def __init__(self, d: int = D_MODEL):
        super().__init__()
        self.geo_proj = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, d),
        )
        self.norm  = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(5.0))   # learned distance scale

    def forward(self, tokens, positions, mask):
        # tokens   : (B, N, D)
        # positions: (B, N, 2)
        # mask     : (B, N) bool — True=padding
        B, N, D = tokens.shape

        diff     = positions.unsqueeze(2) - positions.unsqueeze(1)   # (B,N,N,2)
        dist     = torch.norm(diff, dim=-1, keepdim=True)            # (B,N,N,1)
        cos_a    = diff[..., :1] / (dist + 1e-6)
        sin_a    = diff[..., 1:] / (dist + 1e-6)
        geo_feat = torch.cat([dist, cos_a, sin_a], dim=-1)           # (B,N,N,3)
        geo_emb  = self.geo_proj(geo_feat)                            # (B,N,N,D)

        # Mask padding agents trong chiều source (dim=2)
        dist_flat = dist.squeeze(-1)                                  # (B,N,N)
        pad_pen   = mask.float().unsqueeze(1) * 1e9                   # (B,1,N)
        attn_w    = F.softmax(
            -dist_flat / (self.scale.abs() + 1e-3) - pad_pen, dim=2
        )                                                             # (B,N,N)

        geo_bias = (attn_w.unsqueeze(-1) * geo_emb).sum(dim=2)       # (B,N,D)

        # Zero-out padding agents ở output
        real     = (~mask).float().unsqueeze(-1)                      # (B,N,1)
        geo_bias = geo_bias * real

        return self.norm(tokens + geo_bias)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL BLOCK  (per-agent, RoPE + FlashAttention)
# ═══════════════════════════════════════════════════════════════════════════
class TemporalBlock(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.dh      = d // n_heads
        self.norm1   = nn.LayerNorm(d)
        self.norm2   = nn.LayerNorm(d)
        self.Wqkv    = nn.Linear(d, 3 * d, bias=False)
        self.Wo      = nn.Linear(d, d,     bias=False)
        self.ff      = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )
        self.drop    = nn.Dropout(dropout)
        self.rope    = RotaryEmbedding(self.dh)

    def forward(self, x):
        B, T, D = x.shape
        res = x
        x   = self.norm1(x)
        qkv = self.Wqkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.dh).transpose(1, 2)
                   for t in qkv]
        q, k = self.rope(q, k, T)
        out  = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x   = res + self.drop(self.Wo(out))
        x   = x   + self.drop(self.ff(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# SOCIAL BLOCK  (cross-agent với key_padding_mask)
# ═══════════════════════════════════════════════════════════════════════════
class SocialBlock(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True, bias=False,
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        res = x
        xn  = self.norm1(x)
        out, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x = res + self.drop(out)
        x = x   + self.drop(self.ff(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# INTERLEAVED ENCODER  (#S6: gated residual per layer)
# ═══════════════════════════════════════════════════════════════════════════
class InterleavedEncoder(nn.Module):
    """
    f_in=7  → HeadingExpander → f_proj=8 → Linear(8, D)
    f_in!=7 → dùng thẳng      → f_proj=f_in → Linear(f_in, D)

    Mỗi layer:
      Temporal (per-agent, RoPE)
      → summary = last timestep
      → RelativeGeometry (#S5 attention-weighted)
      → Social (cross-agent, mask)
      → x += alpha.abs() * social_summary  (#S6 gated)
    """
    def __init__(self, f_in: int, d=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        if f_in == 7:
            self.heading_exp = HeadingExpander()
            f_proj = 8
        else:
            self.heading_exp = nn.Identity()
            f_proj = f_in

        self.proj    = nn.Linear(f_proj, d)
        self.norm_in = nn.LayerNorm(d)

        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.social_blocks = nn.ModuleList(
            [SocialBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.geo_enc = RelativeGeometryEncoder(d)

        # Per-layer gating scalar, init = 0.1
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)]
        )

    def forward(self, hist, mask):
        B, N, T, _ = hist.shape

        x = self.heading_exp(hist)       # (B, N, T, f_proj)
        x = self.norm_in(self.proj(x))   # (B, N, T, D)

        positions = hist[:, :, -1, :2]  # (B, N, 2) — last observed xy

        for t_blk, s_blk, alpha in zip(
            self.temporal_blocks, self.social_blocks, self.alphas
        ):
            x_flat = x.view(B * N, T, -1)
            x_flat = t_blk(x_flat)
            x      = x_flat.view(B, N, T, -1)

            summary = x[:, :, -1, :]                         # (B, N, D)
            summary = self.geo_enc(summary, positions, mask)  # geometry
            summary = s_blk(summary, mask)                    # social

            x = x + alpha.abs() * summary.unsqueeze(2)       # gated broadcast

        return x[:, :, -1, :]   # (B, N, D)


# ═══════════════════════════════════════════════════════════════════════════
# PARALLEL DECODER  (K modes, non-autoregressive)
# ═══════════════════════════════════════════════════════════════════════════
class ParallelDecoder(nn.Module):
    def __init__(self, d=D_MODEL, k=K, t_future=T_FUTURE,
                 n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
                 n_layers=2):
        super().__init__()
        self.k        = k
        self.t_future = t_future
        self.d        = d

        self.mode_emb = nn.Parameter(torch.randn(k, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(t_future, d) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.dec       = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.conf_head = nn.Sequential(
            nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, memory, memory_mask=None):
        B = memory.size(0)
        queries = (
            self.mode_emb.unsqueeze(1) + self.time_emb.unsqueeze(0)
        ).view(1, self.k * self.t_future, self.d).expand(B, -1, -1)

        dec     = self.dec(queries, memory,
                           memory_key_padding_mask=memory_mask)
        dec     = dec.view(B, self.k, self.t_future, self.d)
        trajs   = self.out_proj(dec)                           # (B,K,T,2)
        logits  = self.conf_head(dec.mean(dim=2)).squeeze(-1)  # (B,K)
        return trajs, logits


# ═══════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════
class MotionLMv5(nn.Module):
    """
    Preprocessing gốc đã ego-normalize → anchor ≈ (0,0).
    Vẫn cộng anchor để đúng nếu normalize không hoàn hảo.
    AV (index 0) là agent cần predict.
    """
    def __init__(self, f_in: int = F_DIM):
        super().__init__()
        self.encoder = InterleavedEncoder(f_in=f_in)
        self.decoder = ParallelDecoder()

    def forward(self, hist, mask):
        memory    = self.encoder(hist, mask)          # (B, N, D)
        anchor    = hist[:, 0, -1, :2]               # (B, 2) — last AV xy
        trajs, logits = self.decoder(memory, mask)
        abs_trajs = trajs + anchor.unsqueeze(1).unsqueeze(2)
        return abs_trajs, logits


# ═══════════════════════════════════════════════════════════════════════════
# EMA  (#E11: per optimizer step)
# ═══════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.clone().detach()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module, original_sd: dict):
        model.load_state_dict(original_sd)


# ═══════════════════════════════════════════════════════════════════════════
# LOSS  (#L9 div=0.02, #L10 temp=1.0, NaN-safe)
# ═══════════════════════════════════════════════════════════════════════════
def soft_wta_loss(pred_trajs, logits, gt,
                  temperature=SOFT_WTA_TEMP, cls_coeff=CLS_COEFF):
    ade_k   = torch.mean(torch.norm(pred_trajs - gt.unsqueeze(1), dim=-1), dim=-1)
    ade_cl  = torch.clamp(ade_k, 0.0, 1e4)
    weights = F.softmax(-ade_cl / temperature, dim=-1).detach()
    reg     = (weights * ade_cl).sum(dim=1).mean()
    winner  = ade_k.argmin(dim=1).detach()
    cls     = F.cross_entropy(torch.clamp(logits, -10.0, 10.0), winner)
    return reg + cls_coeff * cls


def diversity_loss(pred_trajs, coeff=DIVERSITY_COEFF, margin=2.0):
    ep   = pred_trajs[:, :, -1, :]
    diff = ep.unsqueeze(2) - ep.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    triu = torch.triu(torch.ones(K, K, device=ep.device, dtype=torch.bool), diagonal=1)
    return coeff * F.relu(margin - dist[:, triu]).mean()


def total_loss(pred_trajs, logits, gt):
    swta = soft_wta_loss(pred_trajs, logits, gt)
    div  = diversity_loss(pred_trajs)
    return swta + div, {"swta": swta.item(), "div": div.item()}


# ═══════════════════════════════════════════════════════════════════════════
# NaN GUARD
# ═══════════════════════════════════════════════════════════════════════════
def nan_guard(loss, pred, logits, hist, parts, step):
    if not torch.isnan(loss):
        return False
    print(f"\n  [NaN@{step}]"
          f"  hist:{torch.isnan(hist).any().item()}"
          f"  pred:{torch.isnan(pred).any().item()}"
          f"  logit:{torch.isnan(logits).any().item()}"
          f"  swta={parts['swta']:.4f}  div={parts['div']:.4f}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# METRICS  (#V12)
# ═══════════════════════════════════════════════════════════════════════════
def min_ade(pred, gt):
    return (torch.mean(torch.norm(pred - gt.unsqueeze(1), dim=-1), dim=-1)
            .min(dim=1).values.mean())

def min_fde(pred, gt):
    return (torch.norm(pred[:, :, -1] - gt[:, -1].unsqueeze(1), dim=-1)
            .min(dim=1).values.mean())

def miss_rate(pred, gt, thr=2.0):
    fde = torch.norm(pred[:, :, -1] - gt[:, -1].unsqueeze(1), dim=-1)
    return (fde.min(dim=1).values > thr).float().mean()

def confidence_ade(pred, logits, gt, top_k: int = 1):
    """minADE trong top-k modes theo confidence.  (#V12)"""
    B, Km, T, _ = pred.shape
    _, idx = logits.softmax(dim=-1).topk(min(top_k, Km), dim=-1)
    idx_e  = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, 2)
    top_p  = pred.gather(1, idx_e)
    return (torch.mean(torch.norm(top_p - gt.unsqueeze(1), dim=-1), dim=-1)
            .min(dim=1).values.mean())


# ═══════════════════════════════════════════════════════════════════════════
# MODEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
def print_model_summary(model):
    print("  +--------------------------------------------------+")
    print("  |  Module                  Params                  |")
    print("  +--------------------------------------------------+")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  |  {name:<24}  {n:>10,}              |")
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("  +--------------------------------------------------+")
    print(f"  |  {'TOTAL':<24}  {tot:>10,}              |")
    print("  +--------------------------------------------------+\n")


# ═══════════════════════════════════════════════════════════════════════════
# CSV LOGGER
# ═══════════════════════════════════════════════════════════════════════════
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init = os.path.exists(path)

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not self._init:
                w.writeheader()
                self._init = True
            w.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN EPOCH  (#E11: EMA per optimizer step)
# ═══════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optim, scaler, warmup_sched, cosine_sched,
                device, epoch, warmup_steps, global_step, ema):
    model.train()
    tot = dict(loss=0., ade=0., fde=0., mr=0., t1=0., n=0, skip=0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pbar = tqdm(loader, desc=f"  Ep {epoch:02d} train", ncols=120, leave=False,
                bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}] {postfix}"))

    for hist, fut, mask, n_vec in pbar:
        hist      = hist.to(device, non_blocking=True)
        fut       = fut.to(device,  non_blocking=True)
        mask      = mask.to(device, non_blocking=True)
        agent_fut = fut[:, 0]   # AV future (index 0)

        optim.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            pred, logits = model(hist, mask)
            loss, parts  = total_loss(pred, logits, agent_fut)

        if nan_guard(loss, pred, logits, hist, parts, global_step):
            tot["skip"] += 1; global_step += 1; continue

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optim)
        scaler.update()

        ema.update(model)   # #E11

        if global_step < warmup_steps:
            warmup_sched.step()
        global_step += 1

        bs = hist.size(0)
        with torch.no_grad():
            a  = min_ade(pred.detach(), agent_fut).item()
            f  = min_fde(pred.detach(), agent_fut).item()
            m  = miss_rate(pred.detach(), agent_fut).item()
            t1 = confidence_ade(pred.detach(), logits.detach(), agent_fut, 1).item()
        tot["loss"] += loss.item() * bs
        tot["ade"]  += a  * bs; tot["fde"] += f  * bs
        tot["mr"]   += m  * bs; tot["t1"]  += t1 * bs
        tot["n"]    += bs

        pbar.set_postfix(
            loss  = f"{loss.item():.3f}", ADE = f"{a:.3f}",
            t1ADE = f"{t1:.3f}", gnorm = f"{grad_norm:.2f}",
            lr    = f"{optim.param_groups[0]['lr']:.1e}",
            skip  = tot["skip"],
        )

    cosine_sched.step()
    n = max(tot["n"], 1)
    return (tot["loss"]/n, tot["ade"]/n, tot["fde"]/n,
            tot["mr"]/n, tot["t1"]/n, tot["skip"], global_step)


# ═══════════════════════════════════════════════════════════════════════════
# EVAL EPOCH
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_epoch(model, loader, device, ema: Optional[EMA] = None):
    orig_sd = None
    if ema is not None:
        orig_sd = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply(model)

    model.eval()
    tot = dict(ade=0., fde=0., mr=0., t1=0., t3=0., n=0)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pbar = tqdm(loader, desc="         val  ", ncols=120, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}")

    for hist, fut, mask, n_vec in pbar:
        hist      = hist.to(device, non_blocking=True)
        fut       = fut.to(device,  non_blocking=True)
        mask      = mask.to(device, non_blocking=True)
        agent_fut = fut[:, 0]

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
            pred, logits = model(hist, mask)

        bs = hist.size(0)
        tot["ade"] += min_ade(pred, agent_fut).item() * bs
        tot["fde"] += min_fde(pred, agent_fut).item() * bs
        tot["mr"]  += miss_rate(pred, agent_fut).item() * bs
        tot["t1"]  += confidence_ade(pred, logits, agent_fut, 1).item() * bs
        tot["t3"]  += confidence_ade(pred, logits, agent_fut, 3).item() * bs
        tot["n"]   += bs
        pbar.set_postfix(minADE=f"{tot['ade']/tot['n']:.3f}",
                         top1  =f"{tot['t1']/tot['n']:.3f}")

    if ema and orig_sd:
        ema.restore(model, orig_sd)

    n = max(tot["n"], 1)
    return (tot["ade"]/n, tot["fde"]/n, tot["mr"]/n,
            tot["t1"]/n, tot["t3"]/n)


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════
def save_ckpt(path, model, optim, ema, epoch, metrics, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = getattr(model, "_orig_mod", model)
    torch.save({"epoch": epoch, "step": step,
                "model": raw.state_dict(), "ema": ema.shadow,
                "optim": optim.state_dict(), **metrics}, path)

def load_ckpt(path, model, optim=None, ema=None):
    ckpt = torch.load(path, map_location="cpu")
    raw  = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["model"])
    if optim and "optim" in ckpt: optim.load_state_dict(ckpt["optim"])
    if ema   and "ema"   in ckpt: ema.shadow = ckpt["ema"]
    ep, step = ckpt.get("epoch", 0), ckpt.get("step", 0)
    print(f"  [Resume] epoch={ep} step={step}"
          f"  val_ade={ckpt.get('val_ade', float('nan')):.3f}")
    return ep, step


# ═══════════════════════════════════════════════════════════════════════════
# ARGS
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="MotionLM v5")
    p.add_argument("--train_cache",  default=TRAIN_CACHE)
    p.add_argument("--val_cache",    default=VAL_CACHE)
    p.add_argument("--ckpt_dir",     default=CKPT_DIR)
    p.add_argument("--log_csv",      default=LOG_CSV)
    p.add_argument("--resume",       default=None)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch",        type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--workers",      type=int,
                   default=min(8, os.cpu_count() or 4))
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--compile",      action="store_true")
    p.add_argument("--sanity_check", action="store_true")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = p.parse_known_args()
    return args


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    best_ckpt = os.path.join(args.ckpt_dir, "motionlm_v5_best.pt")
    last_ckpt = os.path.join(args.ckpt_dir, "motionlm_v5_last.pt")

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  MotionLM v5  |  device={device}  K={K}"
          f"  epochs={args.epochs}  batch={args.batch}")
    print(f"  D={D_MODEL}  heads={N_HEADS}  layers={N_LAYERS}  D_FF={D_FF}")
    print(f"  lr={args.lr}  warmup={args.warmup_steps}"
          f"  div={DIVERSITY_COEFF}  temp={SOFT_WTA_TEMP}")
    print(f"  Cache: float16→float32  ego-normalized  F_DIM=auto-detect")
    print(f"  Improvements: AttnGeo(#S5) GatedSocial(#S6) EMAstep(#E11)"
          f" ConfADE(#V12)")
    print(f"{sep}\n")

    # ── [1] Datasets ──────────────────────────────────────────────────────
    print("── [1/4] Datasets " + "─" * 54)
    train_ds = CachedMotionDataset(args.train_cache, args.sanity_check)
    val_ds   = CachedMotionDataset(args.val_cache,   args.sanity_check)

    # F_DIM từ cache thực tế (7 = gốc, 8 = sau expand, 10 = v5 preprocessing)
    f_in = train_ds._f_dim
    expand_note = "→ HeadingExpander (scalar→cos/sin)" if f_in == 7 else "→ pass-through"
    print(f"  Auto-detected F_DIM = {f_in}  {expand_note}\n")

    w     = args.workers
    dl_kw = dict(
        collate_fn         = collate_fn,
        num_workers        = w,
        pin_memory         = (device.type == "cuda"),
        persistent_workers = (w > 0),
        prefetch_factor    = 4 if w > 0 else None,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True, drop_last=True, **dl_kw)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, **dl_kw)
    print(f"  train: {len(train_dl):,} batches  |  val: {len(val_dl):,} batches\n")

    # ── [2] Model ──────────────────────────────────────────────────────────
    print("── [2/4] Model " + "─" * 57)
    model = MotionLMv5(f_in=f_in).to(device)
    print_model_summary(model)

    if args.compile:
        try:
            model = torch.compile(model, fullgraph=False)
            print("  torch.compile: enabled\n")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})\n")

    ema    = EMA(model)
    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.GradScaler(device.type, enabled=(device.type == "cuda"))

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=1e-6,
    )

    start_ep, global_step = 1, 0
    if args.resume and os.path.exists(args.resume):
        print(f"── [2.5] Resume: {args.resume}")
        start_ep, global_step = load_ckpt(args.resume, model, optim, ema)
        start_ep += 1
        print()

    logger   = CSVLogger(args.log_csv)
    best_ade = float("inf")

    # ── [3] Training ──────────────────────────────────────────────────────
    print("── [3/4] Training " + "─" * 54)
    W   = 7
    hdr = (f"  {'Ep':>4} | {'Loss':>{W}} |"
           f" {'trADE':>{W}} {'trFDE':>{W}} {'trMR':>{W}} {'tr1ADE':>{W}} |"
           f" {'vaADE':>{W}} {'vaFDE':>{W}} {'vaMR':>{W}}"
           f" {'va1ADE':>{W}} {'va3ADE':>{W}} |"
           f" {'Skip':>4} {'Time':>6}")
    line = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(line)

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()

        (tr_loss, tr_ade, tr_fde, tr_mr, tr_t1,
         nan_skip, global_step) = train_epoch(
            model, train_dl, optim, scaler,
            warmup_sched, cosine_sched,
            device, ep, args.warmup_steps, global_step, ema,
        )

        val_ade, val_fde, val_mr, val_t1, val_t3 = eval_epoch(
            model, val_dl, device, ema,
        )

        elapsed = time.time() - t0
        lr      = optim.param_groups[0]["lr"]

        row = (f"  {ep:4d} | {tr_loss:{W}.4f} |"
               f" {tr_ade:{W}.3f} {tr_fde:{W}.3f} {tr_mr:{W}.3f} {tr_t1:{W}.3f} |"
               f" {val_ade:{W}.3f} {val_fde:{W}.3f} {val_mr:{W}.3f}"
               f" {val_t1:{W}.3f} {val_t3:{W}.3f} |"
               f" {nan_skip:4d} {elapsed:5.1f}s")

        metrics = dict(val_ade=val_ade, val_fde=val_fde, val_mr=val_mr,
                       val_top1=val_t1, val_top3=val_t3)
        save_ckpt(last_ckpt, model, optim, ema, ep, metrics, global_step)

        logger.log({"epoch": ep, "lr": f"{lr:.2e}",
                    "tr_loss": f"{tr_loss:.4f}",
                    "tr_ade":  f"{tr_ade:.3f}", "tr_fde": f"{tr_fde:.3f}",
                    "tr_mr":   f"{tr_mr:.3f}",  "tr_t1":  f"{tr_t1:.3f}",
                    "val_ade": f"{val_ade:.3f}", "val_fde": f"{val_fde:.3f}",
                    "val_mr":  f"{val_mr:.3f}",
                    "val_t1":  f"{val_t1:.3f}",  "val_t3": f"{val_t3:.3f}",
                    "nan_skip": nan_skip, "time_s": f"{elapsed:.1f}"})

        if val_ade < best_ade:
            best_ade = val_ade
            save_ckpt(best_ckpt, model, optim, ema, ep, metrics, global_step)
            print(row + "  ← best")
        else:
            print(row)

    print(line)
    print(f"\n── [4/4] Done " + "─" * 59)
    print(f"  Best val minADE@{K} : {best_ade:.3f}")
    print(f"  Best ckpt : {best_ckpt}")
    print(f"  Log CSV   : {args.log_csv}\n")


if __name__ == "__main__":
    main()
