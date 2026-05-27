from flask import Flask, request, jsonify, make_response
import threading, os, math, io, csv as csv_mod
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import nest_asyncio, time
from pyngrok import ngrok, conf

nest_asyncio.apply()

# ── Cấu hình ──────────────────────────────────────────────────────────────
PORT            = 5000
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "...........")
DEFAULT_CKPT    = os.getenv("MOTIONLM_CKPT", ".............")

T_HIST   = 20
T_FUTURE = 30
N_MAX    = 3
F_DIM    = 7
K        = 6
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 256
DROPOUT  = 0.1
DT       = 0.1

app = Flask(__name__)
_model      = None
_device     = torch.device("cpu")
_model_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════
class HeadingExpander(nn.Module):
    def forward(self, x):
        h = x[..., 5:6]
        return torch.cat([x[..., :5], torch.cos(h), torch.sin(h), x[..., 6:]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
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
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class RelativeGeometryEncoder(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.geo_proj = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d))
        self.norm  = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, tokens, positions, mask):
        diff     = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist     = torch.norm(diff, dim=-1, keepdim=True)
        cos_a    = diff[..., :1] / (dist + 1e-6)
        sin_a    = diff[..., 1:] / (dist + 1e-6)
        geo_emb  = self.geo_proj(torch.cat([dist, cos_a, sin_a], dim=-1))
        pad_pen  = mask.float().unsqueeze(1) * 1e9
        attn_w   = F.softmax(-dist.squeeze(-1) / (self.scale.abs() + 1e-3) - pad_pen, dim=2)
        geo_bias = (attn_w.unsqueeze(-1) * geo_emb).sum(dim=2)
        return self.norm(tokens + geo_bias * (~mask).float().unsqueeze(-1))


class TemporalBlock(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.n_heads = n_heads
        self.dh      = d // n_heads
        self.norm1   = nn.LayerNorm(d)
        self.norm2   = nn.LayerNorm(d)
        self.Wqkv    = nn.Linear(d, 3 * d, bias=False)
        self.Wo      = nn.Linear(d, d, bias=False)
        self.ff      = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop    = nn.Dropout(dropout)
        self.rope    = RotaryEmbedding(self.dh)

    def forward(self, x):
        B, T, D = x.shape
        res = x
        x   = self.norm1(x)
        qkv = self.Wqkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.dh).transpose(1, 2) for t in qkv]
        q, k = self.rope(q, k, T)
        out  = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        x    = res + self.drop(self.Wo(out))
        return x + self.drop(self.ff(self.norm2(x)))


class SocialBlock(nn.Module):
    def __init__(self, d=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        res = x
        xn  = self.norm1(x)
        out, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x = res + self.drop(out)
        return x + self.drop(self.ff(self.norm2(x)))


class InterleavedEncoder(nn.Module):
    def __init__(self, f_in=F_DIM, d=D_MODEL, n_heads=N_HEADS,
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
            [TemporalBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.social_blocks = nn.ModuleList(
            [SocialBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.geo_enc = RelativeGeometryEncoder(d)
        self.alphas  = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)])

    def forward(self, hist, mask):
        B, N, T, _ = hist.shape
        x = self.norm_in(self.proj(self.heading_exp(hist)))
        positions = hist[:, :, -1, :2]
        for t_blk, s_blk, alpha in zip(self.temporal_blocks, self.social_blocks, self.alphas):
            x_flat = t_blk(x.view(B * N, T, -1))
            x = x_flat.view(B, N, T, -1)
            summary = self.geo_enc(x[:, :, -1, :], positions, mask)
            summary = s_blk(summary, mask)
            x = x + alpha.abs() * summary.unsqueeze(2)
        return x[:, :, -1, :]


class ParallelDecoder(nn.Module):
    def __init__(self, d=D_MODEL, k=K, t_future=T_FUTURE,
                 n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT, n_layers=2):
        super().__init__()
        self.k        = k
        self.t_future = t_future
        self.d        = d
        self.mode_emb = nn.Parameter(torch.randn(k, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(t_future, d) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.dec       = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        self.conf_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, memory, memory_mask=None):
        B = memory.size(0)
        queries = (self.mode_emb.unsqueeze(1) + self.time_emb.unsqueeze(0)
                   ).view(1, self.k * self.t_future, self.d).expand(B, -1, -1)
        dec    = self.dec(queries, memory, memory_key_padding_mask=memory_mask)
        dec    = dec.view(B, self.k, self.t_future, self.d)
        trajs  = self.out_proj(dec)
        logits = self.conf_head(dec.mean(dim=2)).squeeze(-1)
        return trajs, logits


class MotionLMv5(nn.Module):
    def __init__(self, f_in=F_DIM):
        super().__init__()
        self.encoder = InterleavedEncoder(f_in=f_in)
        self.decoder = ParallelDecoder()

    def forward(self, hist, mask):
        memory        = self.encoder(hist, mask)
        anchor        = hist[:, 0, -1, :2]
        trajs, logits = self.decoder(memory, mask)
        return trajs + anchor.unsqueeze(1).unsqueeze(2), logits


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def interp_traj(xy: np.ndarray, target_len: int) -> np.ndarray:
    n = len(xy)
    if n == target_len:
        return xy.astype(np.float32)
    t = np.linspace(0, n - 1, target_len)
    s = np.arange(n)
    return np.stack([np.interp(t, s, xy[:, 0]),
                     np.interp(t, s, xy[:, 1])], axis=1).astype(np.float32)


def get_kinematics(xy: np.ndarray) -> np.ndarray:
    dx      = np.gradient(xy[:, 0])
    dy      = np.gradient(xy[:, 1])
    speed   = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    return np.column_stack([xy[:, 0], xy[:, 1], dx, dy, speed, heading])


def ego_normalize_hist(hist: np.ndarray):
    """
    Ego-normalize history array.
    hist: shape (N, T, F_DIM)
    Returns: hist_norm, anchor_xy, anchor_heading
    """
    rx = float(hist[0, -1, 0])
    ry = float(hist[0, -1, 1])
    rh = float(hist[0, -1, 5])
    c, s = math.cos(-rh), math.sin(-rh)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    h = hist.copy()
    h[:, :, :2]  -= [rx, ry]
    h[:, :, :2]   = h[:, :, :2]  @ R.T
    h[:, :, 2:4]  = h[:, :, 2:4] @ R.T
    h[:, :, 5]   -= rh
    return h, np.array([rx, ry], dtype=np.float32), rh


def ego_denormalize(trajs_norm: np.ndarray, anchor_xy: np.ndarray,
                    anchor_heading: float) -> np.ndarray:
    c, s   = math.cos(anchor_heading), math.sin(anchor_heading)
    R_inv  = np.array([[c, -s], [s, c]], dtype=np.float32)
    out    = trajs_norm.copy()
    out[..., :2] = out[..., :2] @ R_inv.T
    out[..., 0] += anchor_xy[0]
    out[..., 1] += anchor_xy[1]
    return out


def assign_semantic_names(predictions: list, hist_norm: list):
    if len(hist_norm) < 2:
        return
    dx  = float(hist_norm[-1][0]) - float(hist_norm[-2][0])
    dz  = float(hist_norm[-1][1]) - float(hist_norm[-2][1])
    ref = math.atan2(dz, dx)
    for p in predictions:
        traj = p["trajectory"]
        if len(traj) < 2:
            continue
        edx   = traj[-1][0] - traj[0][0]
        edz   = traj[-1][1] - traj[0][1]
        delta = ((math.atan2(edz, edx) - ref) + math.pi) % (2 * math.pi) - math.pi
        p["mode_name"] = "Turn Right" if delta < -0.25 else \
                         "Turn Left"  if delta >  0.25 else "Straight"


# ═══════════════════════════════════════════════════════════════════════════
# PARSE CSV  (chỉ lấy history, không cần future)
# ═══════════════════════════════════════════════════════════════════════════
def parse_csv_to_hist(text: str) -> dict:
    """
    Đọc CSV Argoverse-style, build hist (N, T_HIST, F_DIM), ego-normalize.
    Trả về dict với ok=True hoặc error=str.
    """
    try:
        reader  = csv_mod.DictReader(io.StringIO(text.strip()))
        headers = [h.strip() for h in (reader.fieldnames or [])]
        if not headers:
            return {"error": "CSV không có header"}

        # Tự động nhận diện tên cột
        def find_col(candidates):
            for c in candidates:
                for h in headers:
                    if h.strip().upper() == c.upper():
                        return h
            return None

        col_time  = find_col(["TIMESTAMP", "TIME", "T", "timestamp"])
        col_track = find_col(["TRACK_ID", "ID", "TRACKID", "AGENT_ID", "track_id"])
        col_x     = find_col(["X", "POS_X", "x"])
        col_y     = find_col(["Y", "POS_Y", "y"])
        col_type  = find_col(["OBJECT_TYPE", "TYPE", "AGENT_TYPE", "object_type"])

        missing = [n for n, c in [
            ("TIMESTAMP", col_time), ("TRACK_ID", col_track),
            ("X", col_x),           ("Y", col_y)
        ] if c is None]
        if missing:
            return {"error": f"Thiếu cột: {missing}. Có các cột: {headers}"}

        tracks = defaultdict(list)
        n_rows  = 0
        for row in reader:
            try:
                tid   = str(row[col_track]).strip()
                if not tid:
                    continue
                x     = float(row[col_x])
                y     = float(row[col_y])
                t_val = float(row[col_time])
                otype = "OTHERS"
                if col_type and col_type in row:
                    otype = str(row[col_type]).upper().strip() or "OTHERS"
                tracks[tid].append({"t": t_val, "x": x, "y": y, "type": otype})
                n_rows += 1
            except (ValueError, KeyError):
                continue

        if not tracks:
            return {"error": "Không đọc được trajectory nào"}

        # Sắp xếp track theo độ dài (dài nhất → ưu tiên)
        sorted_tracks = sorted(tracks.items(), key=lambda kv: len(kv[1]), reverse=True)

        # Phân slot: AV, Agent1, Agent2
        slots    = {}
        av_done  = False
        a1_done  = False
        other_n  = 2
        for tid, pts in sorted_tracks:
            if not pts:
                continue
            pts.sort(key=lambda p: p["t"])
            otype = pts[0]["type"]
            if otype == "AV" and not av_done:
                key    = "AV"
                av_done = True
            elif otype == "AGENT" and not a1_done:
                key    = "Agent1"
                a1_done = True
            else:
                if not av_done:
                    key    = "AV"
                    av_done = True
                elif not a1_done:
                    key    = "Agent1"
                    a1_done = True
                elif other_n <= N_MAX:
                    key    = f"Agent{other_n}"
                    other_n += 1
                else:
                    continue
            slots[key] = pts

        if "AV" not in slots:
            return {"error": "Không tìm thấy AV trong CSV"}

        ordered_keys = ["AV"] + [k for k in slots if k != "AV"]
        N = min(len(ordered_keys), N_MAX)

        # Build hist array
        hist_np = np.zeros((N, T_HIST, F_DIM), dtype=np.float32)
        for i, key in enumerate(ordered_keys[:N]):
            pts    = slots[key]
            raw_xy = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
            n_k    = len(raw_xy)
            take   = min(n_k, T_HIST)
            xy     = interp_traj(raw_xy[:take], T_HIST) if take < T_HIST else raw_xy[:T_HIST]
            if n_k >= 2:
                kin = get_kinematics(xy)
            else:
                kin = np.zeros((T_HIST, 6), dtype=np.float32)
            hist_np[i, :, :6] = kin
            hist_np[i, :,  6] = i  # type_id

        # Ego-normalize
        hist_norm, anchor_xy, anchor_heading = ego_normalize_hist(hist_np)

        # Build output per agent
        agents_out = {}
        for i, key in enumerate(ordered_keys[:N]):
            h_pts  = hist_norm[i, :, :2].tolist()
            speeds = hist_norm[i, :,  4].tolist()
            agents_out[key] = {
                "hist": h_pts,
                "stats": {
                    "n_frames": T_HIST,
                    "avg_speed": round(float(np.mean(np.abs(speeds))), 4),
                    "max_speed": round(float(np.max(np.abs(speeds))), 4),
                }
            }

        return {
            "ok":               True,
            "agents":           agents_out,
            "n_agents":         N,
            "agent_keys":       ordered_keys[:N],
            "anchor":           {"xy": anchor_xy.tolist(), "heading": float(anchor_heading)},
            "total_rows":       n_rows,
            "detected_columns": headers,
            # internal use
            "_hist_norm":       hist_norm.tolist(),
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(hist_norm_list: list, anchor_xy: list,
                  anchor_heading: float) -> dict:
    global _model, _device
    with _model_lock:
        if _model is None:
            return {"error": "Model chưa được load"}
        model = _model

    N = min(len(hist_norm_list), N_MAX)

    hist_np = np.zeros((1, N_MAX, T_HIST, F_DIM), dtype=np.float32)
    mask_np = np.ones((1, N_MAX), dtype=bool)
    for i in range(N):
        hist_np[0, i] = np.array(hist_norm_list[i], dtype=np.float32)
        mask_np[0, i] = False

    hist_t = torch.from_numpy(hist_np).to(_device)
    mask_t = torch.from_numpy(mask_np).to(_device)

    pred, logits = model(hist_t, mask_t)
    pred   = pred.float().cpu().numpy()
    confs  = logits.float().softmax(dim=-1)[0].cpu().numpy()

    anc_xy   = np.array(anchor_xy, dtype=np.float32)
    pred_abs = ego_denormalize(pred[0], anc_xy, anchor_heading)

    predictions = []
    for k in range(pred_abs.shape[0]):
        traj      = pred_abs[k].tolist()
        speed_seq = [
            round(math.hypot(traj[i][0] - traj[i-1][0],
                             traj[i][1] - traj[i-1][1]) / DT, 4)
            if i > 0 else 0.0
            for i in range(len(traj))
        ]
        predictions.append({
            "mode":       k,
            "mode_name":  f"Mode-{k}",
            "confidence": round(float(confs[k]), 6),
            "trajectory": traj,
            "speed_seq":  speed_seq,
            "avg_speed":  round(float(np.mean(speed_seq[1:])), 4) if len(speed_seq) > 1 else 0.0,
            "max_speed":  round(float(np.max(speed_seq)), 4),
            "total_dist": round(sum(
                math.hypot(traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1])
                for i in range(1, len(traj))), 4),
        })

    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    assign_semantic_names(predictions, hist_norm_list[0] if hist_norm_list else [])
    return {"predictions": predictions}


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADER
# ═══════════════════════════════════════════════════════════════════════════
def load_checkpoint(ckpt_path: str, device: torch.device) -> MotionLMv5:
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = MotionLMv5(f_in=F_DIM).to(device)
    sd    = ckpt.get("model", ckpt)
    sd_clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd_clean, strict=False)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CORS MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>",             methods=["OPTIONS"])
def options_handler(path):
    return make_response("", 204)


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

HTML ="""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MotionLM v5 — Trajectory Prediction</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');
  :root {
    --bg:#040810;--bg2:#080f1c;--bg3:#0d1829;--panel:#0b1422;
    --border:#1a2d4a;--border2:#243d5e;--accent:#00d4ff;--accent2:#0088cc;
    --green:#00ff9d;--orange:#ff8c42;--red:#ff4466;--yellow:#ffd700;--purple:#9d6fff;
    --text:#c8dff5;--text2:#7a9bb5;--text3:#3d5a73;--radius:6px;
    --glow:0 0 20px rgba(0,212,255,0.15);--glow-s:0 0 10px rgba(0,212,255,0.2);
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  html,body{height:100%;overflow:hidden;}
  body{font-family:'JetBrains Mono',monospace;background:var(--bg);color:var(--text);display:flex;flex-direction:column;}

  header{display:flex;align-items:center;gap:20px;padding:0 24px;height:52px;background:var(--bg2);border-bottom:1px solid var(--border);flex-shrink:0;z-index:10;}
  .logo{font-family:'Syne',sans-serif;font-weight:800;font-size:16px;letter-spacing:0.08em;color:var(--accent);text-shadow:0 0 20px rgba(0,212,255,0.5);white-space:nowrap;}
  .logo span{color:var(--text2);font-weight:400;}
  .header-sep{width:1px;height:24px;background:var(--border);}
  .url-row{display:flex;align-items:center;gap:8px;flex:1;}
  .url-label{font-size:10px;color:var(--text3);white-space:nowrap;text-transform:uppercase;letter-spacing:0.1em;}
  #backendUrl{flex:1;max-width:320px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:11px;padding:5px 10px;outline:none;transition:border-color 0.2s;}
  #backendUrl:focus{border-color:var(--accent2);}
  .btn{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border:1px solid var(--border2);border-radius:var(--radius);background:var(--bg3);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:11px;cursor:pointer;transition:all 0.15s;white-space:nowrap;}
  .btn:hover{border-color:var(--accent);color:var(--accent);box-shadow:var(--glow-s);}
  .btn.primary{background:rgba(0,136,204,0.15);border-color:var(--accent2);color:var(--accent);}
  .btn.primary:hover{background:rgba(0,136,204,0.25);box-shadow:0 0 16px rgba(0,212,255,0.3);}
  .status-dot{width:7px;height:7px;border-radius:50%;background:var(--text3);transition:background 0.3s,box-shadow 0.3s;}
  .status-dot.ok{background:var(--green);box-shadow:0 0 8px rgba(0,255,157,0.6);}
  .status-dot.err{background:var(--red);}
  .status-dot.loading{background:var(--yellow);animation:pulse 1s ease-in-out infinite;}
  #statusText{font-size:10px;color:var(--text2);}
  .model-badge{font-size:10px;color:var(--text3);padding:3px 8px;border:1px solid var(--border);border-radius:3px;}
  .model-badge.loaded{color:var(--green);border-color:rgba(0,255,157,0.3);}
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.4;}}

  .app-body{display:flex;flex:1;overflow:hidden;}

  .left-panel{width:280px;min-width:280px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
  .panel-header{padding:12px 16px;border-bottom:1px solid var(--border);font-size:10px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:var(--text3);display:flex;align-items:center;justify-content:space-between;}
  .panel-header .count{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:1px 8px;font-size:10px;color:var(--text2);}
  .drop-zone{margin:12px;border:1.5px dashed var(--border2);border-radius:var(--radius);padding:20px 12px;text-align:center;cursor:pointer;transition:all 0.2s;position:relative;}
  .drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);background:rgba(0,212,255,0.04);}
  .drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;}
  .drop-icon{font-size:24px;margin-bottom:6px;}
  .drop-title{font-size:11px;color:var(--text);margin-bottom:3px;}
  .drop-sub{font-size:10px;color:var(--text3);}
  .file-list{flex:1;overflow-y:auto;padding:0 8px 8px;}
  .file-list::-webkit-scrollbar{width:4px;}
  .file-list::-webkit-scrollbar-track{background:transparent;}
  .file-list::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
  .file-item{display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:var(--radius);cursor:pointer;border:1px solid transparent;transition:all 0.15s;margin-bottom:2px;}
  .file-item:hover{background:var(--bg3);border-color:var(--border);}
  .file-item.active{background:rgba(0,136,204,0.12);border-color:rgba(0,212,255,0.3);}
  .file-item .fi-icon{width:26px;height:26px;border-radius:4px;background:var(--bg3);display:flex;align-items:center;justify-content:center;font-size:11px;flex-shrink:0;border:1px solid var(--border);}
  .file-item.active .fi-icon{border-color:var(--accent2);color:var(--accent);}
  .fi-info{flex:1;min-width:0;}
  .fi-name{font-size:10px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
  .fi-meta{font-size:9px;color:var(--text3);margin-top:2px;}
  .fi-status{width:6px;height:6px;border-radius:50%;background:var(--text3);flex-shrink:0;}
  .fi-status.ok{background:var(--green);}
  .fi-status.err{background:var(--red);}
  .fi-status.loading{background:var(--yellow);animation:pulse 1s infinite;}
  .panel-actions{padding:10px 12px;border-top:1px solid var(--border);display:flex;gap:6px;}
  .panel-actions .btn{flex:1;justify-content:center;font-size:10px;}

  .center-area{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative;}
  .canvas-toolbar{display:flex;align-items:center;gap:8px;padding:8px 14px;border-bottom:1px solid var(--border);background:var(--bg2);flex-shrink:0;}
  .canvas-toolbar .section-title{font-family:'Syne',sans-serif;font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.1em;}
  .spacer-h{flex:1;}
  .view-btn-group{display:flex;gap:4px;}
  .vbtn{padding:3px 10px;border:1px solid var(--border);border-radius:3px;background:transparent;color:var(--text3);font-family:'JetBrains Mono',monospace;font-size:10px;cursor:pointer;transition:all 0.15s;}
  .vbtn:hover{color:var(--text);border-color:var(--border2);}
  .vbtn.active{background:rgba(0,136,204,0.15);color:var(--accent);border-color:var(--accent2);}

  #canvasContainer{flex:1;position:relative;background:radial-gradient(ellipse at 50% 60%,#06111e 0%,#020609 100%);overflow:hidden;}
  #threeCanvas{width:100%!important;height:100%!important;display:block;}

  /* ══ TRAJECTORY LABEL OVERLAY ══════════════════════════════════
     Label nổi trên đầu trajectory, project từ 3D sang 2D mỗi frame
  ══════════════════════════════════════════════════════════════ */
  #labelOverlay{
    position:absolute;inset:0;
    pointer-events:none;
    z-index:4;
  }
  .traj-label{
    position:absolute;
    transform:translate(-50%,-100%);
    margin-top:-6px;
    padding:3px 9px;
    background:rgba(4,8,16,0.90);
    border:1px solid;
    border-radius:4px;
    font-family:'JetBrains Mono',monospace;
    font-size:10px;
    font-weight:600;
    letter-spacing:0.05em;
    white-space:nowrap;
    pointer-events:none;
    backdrop-filter:blur(6px);
    transition:opacity 0.15s;
    display:none;
  }
  .traj-label.sel{
    font-size:11px;
    padding:4px 11px;
    border-width:1.5px;
    box-shadow:0 2px 16px rgba(0,0,0,0.7);
    z-index:5;
  }
  .traj-label .li{margin-right:5px;}

  .hud-topleft{position:absolute;top:12px;left:12px;display:flex;flex-direction:column;gap:6px;pointer-events:none;}
  .hud-badge{background:rgba(4,8,16,0.85);border:1px solid var(--border);border-radius:4px;padding:5px 10px;font-size:10px;color:var(--text2);backdrop-filter:blur(8px);}
  .hud-badge .val{color:var(--accent);font-weight:600;}
  .hud-badge .ok-val{color:var(--green);}
  .hud-badge .warn-val{color:var(--orange);}
  .hud-topright{position:absolute;top:12px;right:12px;pointer-events:none;display:flex;flex-direction:column;gap:5px;align-items:flex-end;}
  .legend-row{display:flex;align-items:center;gap:7px;background:rgba(4,8,16,0.82);border:1px solid var(--border);border-radius:4px;padding:4px 10px;font-size:10px;color:var(--text2);backdrop-filter:blur(8px);}
  .legend-dot{width:8px;height:8px;border-radius:50%;}
  .hud-bottomleft{position:absolute;bottom:12px;left:12px;font-size:9px;color:var(--text3);pointer-events:none;}
  .hud-bottomright{position:absolute;bottom:12px;right:12px;pointer-events:none;}

  .canvas-overlay{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;background:rgba(4,8,16,0.9);z-index:5;pointer-events:none;opacity:0;transition:opacity 0.3s;}
  .canvas-overlay.show{opacity:1;pointer-events:all;}
  .loader-ring{width:40px;height:40px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .overlay-msg{font-size:11px;color:var(--text2);}

  .empty-state{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;color:var(--text3);pointer-events:none;}
  .empty-state.hidden{display:none;}
  .empty-icon{font-size:40px;opacity:0.3;}
  .empty-title{font-family:'Syne',sans-serif;font-size:14px;color:var(--text3);}
  .empty-sub{font-size:10px;}

  .right-panel{width:300px;min-width:300px;background:var(--panel);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}
  .info-scroll{flex:1;overflow-y:auto;padding:10px;}
  .info-scroll::-webkit-scrollbar{width:4px;}
  .info-scroll::-webkit-scrollbar-track{background:transparent;}
  .info-scroll::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
  .info-block{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);padding:10px;margin-bottom:8px;}
  .info-block-title{font-size:9px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:var(--text3);margin-bottom:8px;display:flex;align-items:center;gap:6px;}
  .info-block-title::after{content:'';flex:1;height:1px;background:var(--border);}
  .kv-row{display:flex;justify-content:space-between;align-items:baseline;font-size:10px;padding:2px 0;}
  .kv-key{color:var(--text3);}
  .kv-val{color:var(--text);font-weight:500;}
  .kv-val.accent{color:var(--accent);}
  .kv-val.green{color:var(--green);}
  .kv-val.orange{color:var(--orange);}
  .agent-card{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:6px;}
  .agent-card-header{display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--bg3);cursor:pointer;border-bottom:1px solid transparent;transition:background 0.15s;}
  .agent-card-header:hover{background:rgba(0,136,204,0.08);}
  .agent-card.open .agent-card-header{border-bottom-color:var(--border);}
  .agent-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
  .agent-name{font-size:11px;font-weight:600;color:var(--text);flex:1;}
  .agent-tag{font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid;text-transform:uppercase;letter-spacing:0.06em;}
  .agent-card-body{display:none;padding:8px 10px;}
  .agent-card.open .agent-card-body{display:block;}
  .mode-list{display:flex;flex-direction:column;gap:4px;margin-top:6px;}
  .mode-item{display:flex;align-items:center;gap:8px;padding:5px 8px;border-radius:4px;border:1px solid var(--border);cursor:pointer;transition:all 0.15s;font-size:10px;}
  .mode-item:hover{border-color:var(--border2);background:var(--bg3);}
  .mode-item.selected{border-color:rgba(0,212,255,0.4);background:rgba(0,136,204,0.1);}
  .mode-swatch{width:8px;height:8px;border-radius:2px;flex-shrink:0;}
  .mode-label{flex:1;color:var(--text);}
  .mode-conf{font-size:9px;color:var(--text3);min-width:34px;text-align:right;}
  .conf-bar-wrap{position:relative;height:3px;background:var(--border);border-radius:2px;width:48px;}
  .conf-bar{position:absolute;left:0;top:0;bottom:0;border-radius:2px;transition:width 0.4s ease;}

  .playback{padding:10px 12px;border-top:1px solid var(--border);background:var(--bg2);flex-shrink:0;}
  .playback-title{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;}
  .pb-controls{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
  .pb-btn{width:28px;height:28px;border-radius:4px;border:1px solid var(--border);background:var(--bg3);color:var(--text2);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;transition:all 0.15s;}
  .pb-btn:hover{border-color:var(--accent);color:var(--accent);}
  .pb-btn.active{background:rgba(0,136,204,0.15);border-color:var(--accent2);color:var(--accent);}
  .pb-time{flex:1;text-align:center;font-size:10px;color:var(--text2);}
  .pb-time span{color:var(--accent);}
  .pb-slider{width:100%;appearance:none;height:3px;background:var(--border);border-radius:2px;outline:none;cursor:pointer;}
  .pb-slider::-webkit-slider-thumb{appearance:none;width:12px;height:12px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px rgba(0,212,255,0.6);cursor:pointer;}

  .toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(60px);background:var(--bg2);border:1px solid var(--border2);border-radius:var(--radius);padding:8px 16px;font-size:11px;color:var(--text);z-index:9999;transition:transform 0.25s ease;pointer-events:none;white-space:nowrap;}
  .toast.show{transform:translateX(-50%) translateY(0);}
  .toast.err{border-color:var(--red);color:var(--red);}
  .toast.ok{border-color:var(--green);color:var(--green);}
</style>
</head>
<body>

<header>
  <div class="logo">MotionLM <span>v5</span></div>
  <div class="header-sep"></div>
  <div class="url-row">
    <span class="url-label">Backend</span>
    <input id="backendUrl" type="text" value="https://runway-parsnip-chariot.ngrok-free.dev/" placeholder="http://localhost:5000" />
    <button class="btn primary" onclick="checkHealth()">⟳ Ping</button>
  </div>
  <div class="status-dot" id="statusDot"></div>
  <span id="statusText">—</span>
  <div class="header-sep"></div>
  <span class="model-badge" id="modelBadge">Model: —</span>
  <button class="btn" onclick="openLoadModel()">⊕ Load Model</button>
</header>

<div class="app-body">

  <div class="left-panel">
    <div class="panel-header">CSV Files<span class="count" id="fileCount">0</span></div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept=".csv" multiple />
      <div class="drop-icon">⬆</div>
      <div class="drop-title">Drop CSV files here</div>
      <div class="drop-sub">or click to browse · multiple files OK</div>
    </div>
    <div class="file-list" id="fileList"></div>
    <div class="panel-actions">
      <button class="btn" onclick="clearFiles()">✕ Clear</button>
      <button class="btn primary" onclick="uploadAll()">▶ Run All</button>
    </div>
  </div>

  <div class="center-area">
    <div class="canvas-toolbar">
      <span class="section-title">3D Scene</span>
      <div class="spacer-h"></div>
      <div class="view-btn-group">
        <button class="vbtn active" id="vbtn3d" onclick="setViewMode('3d')">3D</button>
        <button class="vbtn" id="vbtnTop" onclick="setViewMode('top')">Top</button>
        <button class="vbtn" id="vbtnFollow" onclick="setViewMode('follow')">Follow</button>
      </div>
      <div style="width:1px;height:18px;background:var(--border);margin:0 4px;"></div>
      <button class="btn" onclick="resetCamera()">⌖ Reset</button>
    </div>

    <div id="canvasContainer">
      <canvas id="threeCanvas"></canvas>

      <!-- ══ LABEL OVERLAY: label nổi trên trajectory, project 3D→2D mỗi frame ══ -->
      <div id="labelOverlay"></div>

      <div class="empty-state" id="emptyState">
        <div class="empty-icon">◈</div>
        <div class="empty-title">No scenario loaded</div>
        <div class="empty-sub">Upload a CSV file to begin</div>
      </div>
      <div class="canvas-overlay" id="canvasLoading">
        <div class="loader-ring"></div>
        <div class="overlay-msg" id="loadingMsg">Processing...</div>
      </div>
      <div class="hud-topleft" id="hudTL"></div>
      <div class="hud-topright" id="hudLegend"></div>
      <div class="hud-bottomleft">Drag to orbit · Scroll to zoom · Right-drag to pan</div>
      <div class="hud-bottomright">
        <div class="legend-row" style="font-size:9px;color:var(--text3);">
          <div class="legend-dot" style="background:#00d4ff"></div> AV (ego)
          <div class="legend-dot" style="background:#ff8c42;margin-left:6px"></div> Agents
        </div>
      </div>
    </div>

    <div class="playback">
      <div class="playback-title">Timeline</div>
      <div class="pb-controls">
        <button class="pb-btn" onclick="stepBack()">◀</button>
        <button class="pb-btn" id="btnPlay" onclick="togglePlay()">▶</button>
        <button class="pb-btn" onclick="stepForward()">▶|</button>
        <div class="pb-time">T = <span id="tDisplay">0</span> / <span id="tMax">0</span>s</div>
        <select id="speedSel" style="background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:monospace;font-size:10px;padding:3px 6px;border-radius:3px;cursor:pointer;">
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
      </div>
      <input type="range" class="pb-slider" id="pbSlider" min="0" max="50" value="0" oninput="scrubTo(+this.value)" />
    </div>
  </div>

  <div class="right-panel">
    <div class="panel-header">Analysis</div>
    <div class="info-scroll" id="infoPanel">
      <div style="text-align:center;padding:30px 0;font-size:10px;color:var(--text3);">Select a file to view details</div>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<div id="loadModelModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:100;align-items:center;justify-content:center;">
  <div style="background:var(--panel);border:1px solid var(--border2);border-radius:8px;padding:24px;width:400px;max-width:90vw;">
    <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:var(--accent);margin-bottom:16px;">Load Checkpoint</div>
    <label style="font-size:10px;color:var(--text3);display:block;margin-bottom:4px;">CHECKPOINT PATH</label>
    <input id="ckptInput" type="text" placeholder="/content/drive/MyDrive/checkpoints/motionlm_v5_best.pt"
      style="width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:4px;color:var(--text);font-family:monospace;font-size:11px;padding:7px 10px;outline:none;margin-bottom:12px;" />
    <label style="font-size:10px;color:var(--text3);display:block;margin-bottom:4px;">DEVICE</label>
    <select id="deviceSel" style="width:100%;background:var(--bg3);border:1px solid var(--border);color:var(--text);font-family:monospace;font-size:11px;padding:7px 10px;border-radius:4px;margin-bottom:16px;">
      <option value="cpu">cpu</option>
      <option value="cuda">cuda</option>
    </select>
    <div style="display:flex;gap:8px;justify-content:flex-end;">
      <button class="btn" onclick="closeLoadModel()">Cancel</button>
      <button class="btn primary" onclick="loadModel()">Load</button>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
let files = [], activeIdx = -1;
let T_HIST = 20, T_FUTURE = 30, DT = 0.1;
let playFrame = 0, playing = false, playTimer = null;
let selectedMode = 0;
let renderer, scene, camera;
let sceneObjects = { grid:null, agents:[], trails:[], futurePaths:[] };
let viewMode = '3d';

// Label world-positions — cập nhật khi build predictions
// [{pos3d: THREE.Vector3, mode, modeName, color, conf}]
let _labelData = [];

const MODE_COLORS = ['#00d4ff','#00ff9d','#ffd700','#ff8c42','#ff4466','#9d6fff'];
const AGENT_COLORS = ['#00d4ff','#ff8c42','#ff4466'];
const TURN_ICONS = { 'Turn Left':'↰', 'Turn Right':'↱', 'Straight':'↑' };

// ═══════════════════════════════════════════════════════════════════════════
// SMOOTH TURN TRAJECTORY
// ─────────────────────────────────────────────────────────────────────────
// Mô hình predict thường rẽ đều từ đầu → cuối, trông không tự nhiên.
// Hàm này tạo displayPoints để:
//   ① đi thẳng (0% → 28%)
//   ② rẽ dần với ease-in-out (28% → 62%)  ← khoảng 1.0–1.5 s ở T_FUTURE=30
//   ③ đi thẳng tiếp (62% → 100%)
// Raw trajectory từ model KHÔNG bị ghi đè.
// ═══════════════════════════════════════════════════════════════════════════
function smoothTurnTrajectory(traj, modeName) {
  const n = traj.length;
  if (n < 6) return traj.map(p => [...p]);

  // Tính tổng góc rẽ giữa heading đầu và cuối
  const headingOf = (a, b) => Math.atan2(b[1]-a[1], b[0]-a[0]);
  const hStart = headingOf(traj[0], traj[3]);
  const hEnd   = headingOf(traj[n-4], traj[n-1]);
  let delta = hEnd - hStart;
  // Normalise [-π, π]
  while (delta >  Math.PI) delta -= 2*Math.PI;
  while (delta < -Math.PI) delta += 2*Math.PI;

  // Nếu gần thẳng (<12°) → trả nguyên, không cần smooth
  if (Math.abs(delta) < 0.21) return traj.map(p => [...p]);

  // Vùng rẽ: frame 28%–62%
  const iA = Math.floor(n * 0.28);   // bắt đầu rẽ
  const iB = Math.floor(n * 0.62);   // kết thúc rẽ
  const arcLen = iB - iA;

  // Tốc độ bước trung bình (giữ tốc độ gốc của từng đoạn)
  function segSpeed(i) {
    if (i <= 0 || i >= n) return 0;
    return Math.hypot(traj[i][0]-traj[i-1][0], traj[i][1]-traj[i-1][1]);
  }

  const result = [];

  // ── Đoạn thẳng trước rẽ: copy raw ──────────────────────────
  for (let i = 0; i < iA; i++) {
    result.push([traj[i][0], traj[i][1]]);
  }

  // ── Đoạn rẽ: tích lũy vị trí theo heading xoay dần ─────────
  // Heading tại mỗi step = hStart + delta * ease(t)
  // ease(t) = smoothstep: 3t²-2t³  (S-curve, nhìn tự nhiên)
  for (let i = iA; i <= iB; i++) {
    const t    = (i - iA) / arcLen;                    // 0→1
    const ease = t * t * (3 - 2 * t);                  // smoothstep
    const h    = hStart + delta * ease;
    const spd  = segSpeed(i);
    if (i === iA) {
      result.push([traj[iA][0], traj[iA][1]]);
    } else {
      const prev = result[result.length - 1];
      result.push([
        prev[0] + Math.cos(h) * spd,
        prev[1] + Math.sin(h) * spd,
      ]);
    }
  }

  // ── Đoạn thẳng sau rẽ: tiếp tục theo hEnd ──────────────────
  for (let i = iB + 1; i < n; i++) {
    const spd  = segSpeed(i);
    const prev = result[result.length - 1];
    result.push([
      prev[0] + Math.cos(hEnd) * spd,
      prev[1] + Math.sin(hEnd) * spd,
    ]);
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// THREE.JS INIT
// ═══════════════════════════════════════════════════════════════════════════
function initThree() {
  const container = document.getElementById('canvasContainer');
  const canvas    = document.getElementById('threeCanvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x040810, 0.006);
  const w = container.clientWidth, h = container.clientHeight;
  camera = new THREE.PerspectiveCamera(55, w/h, 0.1, 1000);
  camera.position.set(0, 22, 30);
  camera.lookAt(0, 0, 0);
  renderer.setSize(w, h);
  scene.add(new THREE.AmbientLight(0x112233, 1.2));
  const dl = new THREE.DirectionalLight(0x4488cc, 0.8); dl.position.set(10,30,20); scene.add(dl);
  const rl = new THREE.DirectionalLight(0x00d4ff, 0.3); rl.position.set(-20,5,-10); scene.add(rl);
  buildGrid();
  setupOrbit();
  window.addEventListener('resize', onResize);
  renderLoop();
}

function buildGrid() {
  if (sceneObjects.grid) scene.remove(sceneObjects.grid);
  const g = new THREE.Group();
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(200,200),
    new THREE.MeshLambertMaterial({color:0x040d18, side:THREE.DoubleSide})
  );
  plane.rotation.x = -Math.PI/2; plane.position.y = -0.02; g.add(plane);
  g.add(new THREE.GridHelper(120, 40, 0x0d2540, 0x091a2e));
  const cm = new THREE.LineBasicMaterial({color:0x1a3d5e});
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-3,0,0),new THREE.Vector3(3,0,0)]),cm));
  g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,-3),new THREE.Vector3(0,0,3)]),cm));
  sceneObjects.grid = g; scene.add(g);
}

let orbit = { theta:0.5, phi:0.9, radius:40, target:new THREE.Vector3(0,0,0) };
let drag   = { active:false, right:false, sx:0, sy:0 };

function setupOrbit() {
  const el = renderer.domElement;
  el.addEventListener('mousedown', e => { drag.active=true; drag.right=e.button===2; drag.sx=e.clientX; drag.sy=e.clientY; e.preventDefault(); });
  window.addEventListener('mouseup', () => { drag.active=false; });
  window.addEventListener('mousemove', e => {
    if (!drag.active) return;
    const dx=(e.clientX-drag.sx)*0.005, dy=(e.clientY-drag.sy)*0.005;
    drag.sx=e.clientX; drag.sy=e.clientY;
    if (drag.right) {
      const right=new THREE.Vector3();
      right.crossVectors(camera.position.clone().sub(orbit.target), new THREE.Vector3(0,1,0)).normalize();
      orbit.target.addScaledVector(right, -dx*orbit.radius*0.3);
      orbit.target.y += dy*orbit.radius*0.3;
    } else {
      orbit.theta -= dx;
      orbit.phi = Math.max(0.1, Math.min(Math.PI*0.48, orbit.phi-dy));
    }
  });
  el.addEventListener('wheel', e => { orbit.radius=Math.max(5,Math.min(150,orbit.radius*(1+e.deltaY*0.001))); });
  el.addEventListener('contextmenu', e => e.preventDefault());
}

function applyOrbit() {
  const x=orbit.target.x+orbit.radius*Math.sin(orbit.phi)*Math.sin(orbit.theta);
  const y=orbit.target.y+orbit.radius*Math.cos(orbit.phi);
  const z=orbit.target.z+orbit.radius*Math.sin(orbit.phi)*Math.cos(orbit.theta);
  camera.position.set(x,y,z); camera.lookAt(orbit.target);
}
function resetCamera() { orbit.theta=0.5; orbit.phi=0.9; orbit.radius=40; orbit.target.set(0,0,0); }

function onResize() {
  const c=document.getElementById('canvasContainer'); const w=c.clientWidth,h=c.clientHeight;
  renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();
}

function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (viewMode==='follow') applyFollowCam();
  else if (viewMode!=='top') applyOrbit();
  renderer.render(scene, camera);
  // Project label 3D→2D mỗi frame
  _projectLabels();
}

function setViewMode(mode) {
  viewMode=mode;
  document.querySelectorAll('.vbtn').forEach(b=>b.classList.remove('active'));
  if(mode==='3d')     document.getElementById('vbtn3d').classList.add('active');
  if(mode==='top')    document.getElementById('vbtnTop').classList.add('active');
  if(mode==='follow') document.getElementById('vbtnFollow').classList.add('active');
  if(mode==='top') { camera.position.set(orbit.target.x,60,orbit.target.z+0.001); camera.lookAt(orbit.target); }
}

function applyFollowCam() {
  if (!sceneObjects.agents.length) { applyOrbit(); return; }
  const av=sceneObjects.agents[0]; if(!av){applyOrbit();return;}
  camera.position.set(av.position.x-8, av.position.y+6, av.position.z+12);
  camera.lookAt(av.position);
}

// Argoverse CSV: X=east, Y=north  →  Three.js: X=right, Y=up, Z=-north
function csvToThree(cx,cy) { return { x:cx, z:-cy }; }

// ═══════════════════════════════════════════════════════════════════════════
// SCENE BUILD
// ═══════════════════════════════════════════════════════════════════════════
function clearScene() {
  [...sceneObjects.agents,...sceneObjects.trails,...sceneObjects.futurePaths].forEach(o=>{if(o&&o.parent)scene.remove(o);});
  sceneObjects.agents=[]; sceneObjects.trails=[]; sceneObjects.futurePaths=[];
  _labelData=[];
  document.getElementById('labelOverlay').innerHTML='';
}

function buildScene(data) {
  clearScene();
  if (!data||!data.agents) return;
  document.getElementById('emptyState').classList.add('hidden');
  updateHUD(data);
  const keys = data.agent_keys||Object.keys(data.agents);
  keys.forEach((key,idx)=>{ const ag=data.agents[key]; if(ag) buildAgentObjects(key,ag,idx,data); });
  updateScene(playFrame);
}

function buildAgentObjects(key, agent, idx, data) {
  const color = new THREE.Color(AGENT_COLORS[Math.min(idx,AGENT_COLORS.length-1)]);
  const isAV  = idx===0;
  const hist  = agent.hist||[];

  let mesh;
  if (isAV) {
    mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1.8,0.6,3.8),
      new THREE.MeshPhongMaterial({color,emissive:color,emissiveIntensity:0.3,transparent:true,opacity:1.0})
    );
    const top = new THREE.Mesh(new THREE.BoxGeometry(1.4,0.1,2.8),new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.4}));
    top.position.y=0.38; mesh.add(top);
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.3,0.8,8),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.6}));
    arrow.position.set(0,0.5,-2.2); arrow.rotation.x=-Math.PI/2; mesh.add(arrow);
  } else {
    mesh = new THREE.Mesh(
      new THREE.BoxGeometry(1.6,0.5,3.2),
      new THREE.MeshPhongMaterial({color,emissive:color,emissiveIntensity:0.2,transparent:true,opacity:0.9})
    );
  }
  mesh.position.y=0.35; mesh.castShadow=true;
  mesh.userData={key,idx,hist,color:color.getHex()};
  scene.add(mesh); sceneObjects.agents.push(mesh);

  const tg=new THREE.Group(); tg.userData={key,hist,T:hist.length};
  scene.add(tg); sceneObjects.trails.push(tg);
  buildTrail(tg, hist, color, isAV);

  // Build predictions hanya sekali untuk AV
  if (isAV && data.agents.AV && data.agents.AV.predictions) {
    buildFuturePaths(data.agents.AV.predictions);
  }
}

function buildTrail(group, hist, color, isAV) {
  if (hist.length<2) return;
  const points = hist.map(([cx,cy])=>{ const{x,z}=csvToThree(cx,cy); return new THREE.Vector3(x,0.15,z); });
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({color,transparent:true,opacity:isAV?0.8:0.5})
  );
  line.userData.allPoints=points; group.add(line);
  points.forEach((p,i)=>{
    if(i%4!==0) return;
    const dot=new THREE.Mesh(new THREE.SphereGeometry(0.08,6,4),new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.5}));
    dot.position.copy(p); dot.userData.frameIdx=i; group.add(dot);
  });
}

// ══════════════════════════════════════════════════════════════════
// BUILD FUTURE PATHS — với smooth trajectory + label overhead
// THAY ĐỔI SO VỚI BẢN GỐC:
//  1. Dùng smoothTurnTrajectory() → đi thẳng–rẽ–thẳng
//  2. Label đặt tại midpoint (50%) của trajectory, y=2.0 (nổi lên)
//  3. Label project 3D→2D trong renderLoop() (_projectLabels)
// ══════════════════════════════════════════════════════════════════
function buildFuturePaths(predictions) {
  _labelData = [];

  predictions.forEach((pred, ki) => {
    const col    = new THREE.Color(MODE_COLORS[ki % MODE_COLORS.length]);
    const traj   = pred.trajectory || [];
    const mName  = pred.mode_name || `Mode-${ki}`;
    if (traj.length < 2) return;

    // Tạo display trajectory đã smooth
    const dispTraj = smoothTurnTrajectory(traj, mName);

    const group  = new THREE.Group();
    group.userData = { mode:ki, conf:pred.confidence, modeName:mName };

    const points = dispTraj.map(([cx,cy]) => {
      const{x,z}=csvToThree(cx,cy);
      return new THREE.Vector3(x,0.2,z);
    });

    // ── Main path line ──
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({color:col,transparent:true,opacity:ki===selectedMode?0.95:0.22});
    const line = new THREE.Line(geo, mat);
    line.userData.isFutureLine=true;
    line.userData.allPoints=points;
    line.userData.mode=ki;
    group.add(line);

    // ── Endpoint sphere ──
    const ep   = points[points.length-1];
    const eMsh = new THREE.Mesh(
      new THREE.SphereGeometry(ki===selectedMode?0.36:0.18,10,6),
      new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:ki===selectedMode?1.0:0.35})
    );
    eMsh.position.copy(ep); eMsh.userData.isFutureEnd=true; group.add(eMsh);

    // ── Direction dots ──
    for (let i=4; i<points.length; i+=5) {
      const d=new THREE.Mesh(
        new THREE.SphereGeometry(0.06,4,4),
        new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:ki===selectedMode?0.65:0.12})
      );
      d.position.copy(points[i]); d.position.y=0.18; group.add(d);
    }

    // ── Label world position = midpoint của trajectory, y = 2.0 ──
    const midIdx  = Math.floor(points.length * 0.50);
    const labelP  = points[midIdx].clone();
    labelP.y = 2.0;   // nổi 2m trên mặt đất

    _labelData.push({
      pos3d:    labelP,
      mode:     ki,
      modeName: mName,
      color:    '#' + col.getHexString(),
      conf:     pred.confidence,
    });

    group.visible = true;
    scene.add(group);
    sceneObjects.futurePaths.push(group);
  });

  // Tạo DOM label elements
  _buildLabelDOM();
}

// ─── Tạo DOM label ─────────────────────────────────────────────────
function _buildLabelDOM() {
  const overlay = document.getElementById('labelOverlay');
  overlay.innerHTML = '';
  _labelData.forEach(lw => {
    const el  = document.createElement('div');
    const sel = lw.mode === selectedMode;
    el.className   = 'traj-label' + (sel ? ' sel' : '');
    el.id          = `tlbl-${lw.mode}`;
    el.style.color        = lw.color;
    el.style.borderColor  = lw.color + (sel ? 'cc' : '44');
    el.style.opacity      = sel ? '1' : '0.55';
    const icon = TURN_ICONS[lw.modeName] || '→';
    el.innerHTML = `<span class="li">${icon}</span>${lw.modeName}<span style="opacity:0.6;font-size:9px;margin-left:5px;">${(lw.conf*100).toFixed(0)}%</span>`;
    overlay.appendChild(el);
  });
}

// ─── Project 3D label positions → 2D screen mỗi render frame ──────
function _projectLabels() {
  if (!renderer || !camera) return;
  const c  = document.getElementById('canvasContainer');
  const W  = c.clientWidth, H = c.clientHeight;
  const inFuture = playFrame >= T_HIST;

  _labelData.forEach(lw => {
    const el = document.getElementById(`tlbl-${lw.mode}`);
    if (!el) return;

    // Chỉ hiện trong future phase
    if (!inFuture) { el.style.display='none'; return; }

    // Project world pos → NDC [-1,1]
    const ndc = lw.pos3d.clone().project(camera);
    // Lọc điểm sau camera hoặc ngoài màn hình
    if (ndc.z > 1) { el.style.display='none'; return; }
    const px = ( ndc.x * 0.5 + 0.5) * W;
    const py = (-ndc.y * 0.5 + 0.5) * H;
    if (px < -80 || px > W+80 || py < -40 || py > H+40) { el.style.display='none'; return; }

    el.style.display = 'block';
    el.style.left    = px + 'px';
    el.style.top     = py + 'px';
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// UPDATE SCENE PER FRAME
// ═══════════════════════════════════════════════════════════════════════════
function updateScene(frame) {
  const histPhase = frame < T_HIST;
  const futFrame  = frame - T_HIST;

  // ── Agents ──────────────────────────────────────────────────
  sceneObjects.agents.forEach((mesh, idx) => {
    const { hist } = mesh.userData;
    const T = hist.length;
    let vf  = Math.min(frame, T-1); if(vf<0)vf=0;

    const [cx,cy] = hist[vf] || [0,0];
    const {x,z}   = csvToThree(cx,cy);
    mesh.position.x = x; mesh.position.z = z;

    // Heading từ lịch sử
    if (vf > 0) {
      const [px,py] = hist[vf-1]||hist[vf];
      const [nx,ny] = hist[vf];
      const dhx=nx-px, dhy=ny-py;
      if(Math.abs(dhx)>0.001||Math.abs(dhy)>0.001)
        mesh.rotation.y = Math.atan2(dhx, dhy);
    }

    if (idx === 0) {
      // AV: luôn hiện; trong future phase, di chuyển theo prediction đã smooth
      mesh.visible = true;
      mesh.material.opacity = 1.0;
      if (!histPhase) {
        const fp = sceneObjects.futurePaths[selectedMode];
        if (fp) {
          const ln = fp.children.find(c=>c.isLine&&c.userData.isFutureLine);
          if (ln && ln.userData.allPoints) {
            const pts = ln.userData.allPoints;
            const fi  = Math.min(futFrame, pts.length-1);
            if (fi >= 0) {
              mesh.position.x = pts[fi].x;
              mesh.position.z = pts[fi].z;
              // Heading theo smooth trajectory
              if (fi > 0) {
                const prev=pts[fi-1], cur=pts[fi];
                const dx=cur.x-prev.x, dz=cur.z-prev.z;
                if(Math.abs(dx)>0.001||Math.abs(dz)>0.001)
                  mesh.rotation.y = Math.atan2(-dz, dx) + Math.PI/2;
              }
            }
          }
        }
      }
    } else {
      // Non-AV: fade out nhanh (4 frame) khi vào future phase
      if (!histPhase) {
        const alpha = Math.max(0, 0.9 - (futFrame/4)*0.9);
        mesh.material.opacity = alpha;
        mesh.visible = alpha > 0.02;
      } else {
        mesh.visible = true; mesh.material.opacity = 0.9;
      }
    }
  });

  // ── Trails: clip đến frame hiện tại ─────────────────────────
  sceneObjects.trails.forEach(group => {
    group.children.forEach(child => {
      if (child.isLine) {
        const all = child.userData.allPoints||[];
        const pts = all.slice(0, Math.max(2, Math.min(frame+1, all.length)));
        if (pts.length >= 2) {
          const pos = new Float32Array(pts.length*3);
          pts.forEach((p,i)=>{pos[i*3]=p.x;pos[i*3+1]=p.y;pos[i*3+2]=p.z;});
          child.geometry.setAttribute('position',new THREE.BufferAttribute(pos,3));
          child.geometry.setDrawRange(0,pts.length);
        }
        child.visible=true;
      } else if (child.isMesh) {
        child.visible = child.userData.frameIdx!==undefined ? child.userData.frameIdx<=frame : true;
      }
    });
  });

  // ── Future paths: ẩn trong hist phase, clip trong future phase ─
  sceneObjects.futurePaths.forEach((group) => {
    if (histPhase) {
      group.visible=false;
    } else {
      group.visible=true;
      group.children.forEach(child=>{
        if (child.isLine && child.userData.isFutureLine) {
          const all=child.userData.allPoints;
          if (!all) return;
          const pts=all.slice(0,Math.max(2,futFrame+2));
          if(pts.length>=2){
            const pos=new Float32Array(pts.length*3);
            pts.forEach((p,i)=>{pos[i*3]=p.x;pos[i*3+1]=p.y;pos[i*3+2]=p.z;});
            child.geometry.setAttribute('position',new THREE.BufferAttribute(pos,3));
            child.geometry.setDrawRange(0,pts.length);
          }
        }
      });
    }
  });

  _updateModeVisuals();
  _updateTimeDisplay(frame);
}

function _updateModeVisuals() {
  sceneObjects.futurePaths.forEach((group,ki)=>{
    const isSel=ki===selectedMode;
    group.children.forEach(child=>{
      if(!child.material) return;
      if(child.userData.isFutureEnd){
        child.material.opacity=isSel?1.0:0.28; child.scale.setScalar(isSel?1.5:0.8);
      } else if(child.isLine){
        child.material.opacity=isSel?0.95:0.18;
      } else {
        child.material.opacity=isSel?0.65:0.10;
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// HUD
// ═══════════════════════════════════════════════════════════════════════════
function updateHUD(data) {
  document.getElementById('hudTL').innerHTML=`
    <div class="hud-badge">Agents:<span class="val"> ${data.n_agents||'?'}</span> &nbsp;·&nbsp; Hist:<span class="val"> ${T_HIST}×${DT}s</span> &nbsp;·&nbsp; Fut:<span class="val"> ${T_FUTURE}×${DT}s</span></div>
    ${data.model_inferred
      ?`<div class="hud-badge">Model: <span class="ok-val">✓ Inferred</span> &nbsp; K=6 modes</div>`
      :`<div class="hud-badge">Model: <span class="warn-val">⚠ Not loaded</span></div>`}`;
  if (data.model_inferred && data.agents?.AV?.predictions) {
    document.getElementById('hudLegend').innerHTML=data.agents.AV.predictions.map((p,i)=>`
      <div class="legend-row">
        <div class="legend-dot" style="background:${MODE_COLORS[i]}"></div>
        ${TURN_ICONS[p.mode_name]||'→'} ${p.mode_name||'Mode-'+i}
        <span style="color:${MODE_COLORS[i]};font-size:9px;margin-left:4px;">${(p.confidence*100).toFixed(1)}%</span>
      </div>`).join('');
  }
}

function _updateTimeDisplay(frame) {
  const total=T_HIST+T_FUTURE;
  document.getElementById('tDisplay').textContent=(frame*DT).toFixed(1);
  document.getElementById('tMax').textContent=((total-1)*DT).toFixed(1);
  document.getElementById('pbSlider').max=total-1;
  document.getElementById('pbSlider').value=frame;
  // Màu slider: grey cho hist, cyan cho future
  const pct = (frame/(total-1))*100;
  const hp  = ((T_HIST-1)/(total-1))*100;
  document.getElementById('pbSlider').style.background=
    `linear-gradient(to right,#1a2d4a 0%,#1a2d4a ${hp}%,rgba(0,136,204,0.45) ${hp}%,rgba(0,212,255,0.7) ${pct}%,#1a2d4a ${pct}%)`;
}

// ═══════════════════════════════════════════════════════════════════════════
// PLAYBACK
// ═══════════════════════════════════════════════════════════════════════════
function togglePlay(){
  playing=!playing;
  document.getElementById('btnPlay').classList.toggle('active',playing);
  document.getElementById('btnPlay').textContent=playing?'⏸':'▶';
  if(playing) tickPlay();
}
function tickPlay(){
  if(!playing) return;
  const total=T_HIST+T_FUTURE;
  const speed=+document.getElementById('speedSel').value;
  playFrame++; if(playFrame>=total)playFrame=0;
  updateScene(playFrame);
  playTimer=setTimeout(tickPlay,DT*1000/speed);
}
function stepBack(){stopPlay();playFrame=Math.max(0,playFrame-1);updateScene(playFrame);}
function stepForward(){stopPlay();playFrame=Math.min(T_HIST+T_FUTURE-1,playFrame+1);updateScene(playFrame);}
function scrubTo(v){stopPlay();playFrame=+v;updateScene(playFrame);}
function stopPlay(){playing=false;clearTimeout(playTimer);document.getElementById('btnPlay').classList.remove('active');document.getElementById('btnPlay').textContent='▶';}

// ═══════════════════════════════════════════════════════════════════════════
// FILE / API
// ═══════════════════════════════════════════════════════════════════════════
async function checkHealth(){
  const dot=document.getElementById('statusDot'),txt=document.getElementById('statusText'),badge=document.getElementById('modelBadge');
  dot.className='status-dot loading';txt.textContent='Connecting...';
  try{
    const r=await fetch(`${getBase()}/api/health`,{signal:AbortSignal.timeout(5000)});
    const d=await r.json();
    dot.className='status-dot ok';txt.textContent=d.device||'ok';
    if(d.model_loaded){badge.textContent='Model: ✓ Loaded';badge.className='model-badge loaded';}
    else{badge.textContent='Model: Not loaded';badge.className='model-badge';}
    toast('Backend connected','ok');
  }catch(e){
    dot.className='status-dot err';txt.textContent='Offline';badge.textContent='Model: —';badge.className='model-badge';
    toast('Cannot reach backend','err');
  }
}
function getBase(){return document.getElementById('backendUrl').value.replace(/\/$/,'');}

document.getElementById('fileInput').addEventListener('change',e=>{addFiles(Array.from(e.target.files));e.target.value='';});
const dz=document.getElementById('dropZone');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag-over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag-over'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag-over');addFiles(Array.from(e.dataTransfer.files).filter(f=>f.name.endsWith('.csv')));});

function addFiles(nf){
  nf.forEach(f=>{if(!files.find(x=>x.name===f.name))files.push({file:f,name:f.name,status:'pending',data:null});});
  renderFileList();
  if(nf.length===1){activeIdx=files.length-1;uploadSingle(activeIdx);}
}
function clearFiles(){
  files=[];activeIdx=-1;renderFileList();clearScene();
  document.getElementById('emptyState').classList.remove('hidden');
  document.getElementById('infoPanel').innerHTML='<div style="text-align:center;padding:30px 0;font-size:10px;color:var(--text3);">Select a file to view details</div>';
  document.getElementById('hudTL').innerHTML='';document.getElementById('hudLegend').innerHTML='';
}
function renderFileList(){
  document.getElementById('fileCount').textContent=files.length;
  document.getElementById('fileList').innerHTML=files.map((f,i)=>`
    <div class="file-item ${i===activeIdx?'active':''}" onclick="selectFile(${i})">
      <div class="fi-icon">📄</div>
      <div class="fi-info">
        <div class="fi-name" title="${f.name}">${f.name}</div>
        <div class="fi-meta">${f.data?(f.data.n_agents||'?')+' agents':f.status==='loading'?'Processing…':'—'}</div>
      </div>
      <div class="fi-status ${f.status==='ok'?'ok':f.status==='err'?'err':f.status==='loading'?'loading':''}"></div>
    </div>`).join('');
}
function selectFile(i){
  activeIdx=i;renderFileList();
  const f=files[i];
  if(f.status==='ok'&&f.data)displayData(f.data);
  else if(f.status==='pending'||f.status==='err')uploadSingle(i);
}
async function uploadAll(){for(let i=0;i<files.length;i++)await uploadSingle(i);}
async function uploadSingle(idx){
  const f=files[idx];f.status='loading';renderFileList();showLoading(true,`Parsing ${f.name}…`);
  try{
    const fd=new FormData();fd.append('files',f.file);
    const r=await fetch(`${getBase()}/api/upload_csv`,{method:'POST',body:fd});
    const d=await r.json();
    if(d.results&&d.results[0]){
      const res=d.results[0];
      if(res.error){f.status='err';f.data=null;toast(`${f.name}: ${res.error}`,'err');}
      else{f.status='ok';f.data=res;if(activeIdx===idx)displayData(res);toast(`${f.name}: OK`,'ok');}
    }else{f.status='err';toast('Unexpected response','err');}
  }catch(e){f.status='err';toast(`Upload failed: ${e.message}`,'err');}
  showLoading(false);renderFileList();
}
function displayData(data){
  stopPlay();playFrame=0;
  buildScene(data);
  updateScene(0);
  renderInfoPanel(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// INFO PANEL
// ═══════════════════════════════════════════════════════════════════════════
function renderInfoPanel(data){
  let html=`<div class="info-block">
    <div class="info-block-title">Scenario</div>
    <div class="kv-row"><span class="kv-key">File</span><span class="kv-val">${(data.filename||'?').replace('.csv','')}</span></div>
    <div class="kv-row"><span class="kv-key">Agents</span><span class="kv-val accent">${data.n_agents}</span></div>
    <div class="kv-row"><span class="kv-key">Total rows</span><span class="kv-val">${data.total_rows}</span></div>
    <div class="kv-row"><span class="kv-key">Inferred</span><span class="kv-val ${data.model_inferred?'green':'orange'}">${data.model_inferred?'✓ Yes':'✗ No model'}</span></div>
    <div class="kv-row"><span class="kv-key">Anchor</span><span class="kv-val">(${data.anchor.xy.map(v=>v.toFixed(1)).join(', ')})</span></div>
  </div>`;
  const keys=data.agent_keys||Object.keys(data.agents);
  keys.forEach((key,idx)=>{
    const ag=data.agents[key];if(!ag)return;
    const col=AGENT_COLORS[Math.min(idx,AGENT_COLORS.length-1)];
    const isAV=idx===0;
    html+=`<div class="agent-card ${isAV?'open':''}">
      <div class="agent-card-header" onclick="toggleCard(this)">
        <div class="agent-dot" style="background:${col}"></div>
        <div class="agent-name">${key}</div>
        <div class="agent-tag" style="color:${col};border-color:${col}40;">${isAV?'EGO':'AGENT'}</div>
        <span style="font-size:10px;color:var(--text3);margin-left:4px;">▾</span>
      </div>
      <div class="agent-card-body">
        <div class="kv-row"><span class="kv-key">Avg speed</span><span class="kv-val">${ag.stats?.avg_speed??'?'} m/s</span></div>
        <div class="kv-row"><span class="kv-key">Max speed</span><span class="kv-val">${ag.stats?.max_speed??'?'} m/s</span></div>
        ${isAV&&ag.predictions?renderPredictions(ag.predictions):''}
      </div>
    </div>`;
  });
  document.getElementById('infoPanel').innerHTML=html;
}
function renderPredictions(preds){
  let html=`<div style="font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:0.12em;margin-top:8px;margin-bottom:6px;">Predicted Modes</div><div class="mode-list">`;
  preds.forEach((p,i)=>{
    const col=MODE_COLORS[i%MODE_COLORS.length];
    const conf=(p.confidence*100).toFixed(1);
    const icon=TURN_ICONS[p.mode_name]||'→';
    html+=`<div class="mode-item ${i===selectedMode?'selected':''}" onclick="selectMode(${i})">
      <div class="mode-swatch" style="background:${col}"></div>
      <div class="mode-label">${icon} ${p.mode_name||'Mode-'+i}</div>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:${conf}%;background:${col}"></div></div>
      <div class="mode-conf">${conf}%</div>
    </div>`;
  });
  html+=`</div>`;
  if(preds[selectedMode]){
    const p=preds[selectedMode];
    html+=`<div style="margin-top:8px;padding:6px 8px;background:var(--bg);border-radius:4px;border:1px solid var(--border);">
      <div class="kv-row"><span class="kv-key">Avg speed</span><span class="kv-val">${p.avg_speed} m/s</span></div>
      <div class="kv-row"><span class="kv-key">Max speed</span><span class="kv-val">${p.max_speed} m/s</span></div>
      <div class="kv-row"><span class="kv-key">Distance</span><span class="kv-val">${p.total_dist} m</span></div>
    </div>`;
  }
  return html;
}
function selectMode(i){
  selectedMode=i;
  _updateModeVisuals();
  // Cập nhật style label DOM
  _labelData.forEach(lw=>{
    const el=document.getElementById(`tlbl-${lw.mode}`);
    if(!el)return;
    const sel=lw.mode===i;
    el.className='traj-label'+(sel?' sel':'');
    el.style.borderColor=lw.color+(sel?'cc':'44');
    el.style.opacity=sel?'1':'0.55';
  });
  const f=files[activeIdx];if(f&&f.data)renderInfoPanel(f.data);
}
function toggleCard(h){h.parentElement.classList.toggle('open');}

// ── Model ──
function openLoadModel(){document.getElementById('loadModelModal').style.display='flex';}
function closeLoadModel(){document.getElementById('loadModelModal').style.display='none';}
async function loadModel(){
  const ckpt=document.getElementById('ckptInput').value;
  const device=document.getElementById('deviceSel').value;
  closeLoadModel();toast('Loading model…');
  try{
    const r=await fetch(`${getBase()}/api/load_model`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ckpt,device})});
    const d=await r.json();
    if(d.error){toast(d.error,'err');return;}
    document.getElementById('modelBadge').textContent='Model: ✓ Loaded';
    document.getElementById('modelBadge').className='model-badge loaded';
    toast(`Model loaded · ${(d.params/1e6).toFixed(2)}M params`,'ok');
  }catch(e){toast('Load failed: '+e.message,'err');}
}

// ── Utils ──
function showLoading(show,msg=''){
  document.getElementById('canvasLoading').classList.toggle('show',show);
  if(msg)document.getElementById('loadingMsg').textContent=msg;
}
let toastTimer;
function toast(msg,type=''){
  const el=document.getElementById('toast');
  el.textContent=msg;el.className='toast show'+(type?' '+type:'');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),2800);
}

window.addEventListener('load',()=>{initThree();checkHealth();});
</script>
</body>
</html>
"""
# ── HTML trống ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if HTML:
        return make_response(HTML, 200, {"Content-Type": "text/html; charset=utf-8"})
    return make_response(
        "<h3 style='font-family:monospace;padding:20px;color:#00d4ff;background:#050c18'>"
        "MotionLM v5 Backend running.<br><br>"
        "Frontend: mở <b>motionlm_frontend.html</b> trong browser.<br>"
        "Set <code>BACKEND_URL</code> trong file HTML trỏ đến server này.</h3>",
        200, {"Content-Type": "text/html; charset=utf-8"}
    )


@app.route("/api/health", methods=["GET"])
def health():
    with _model_lock:
        loaded = _model is not None
    return jsonify({
        "status":       "ok",
        "model_loaded": loaded,
        "device":       str(_device),
        "t_hist":       T_HIST,
        "t_future":     T_FUTURE,
        "k_modes":      K,
    })


@app.route("/api/load_model", methods=["POST"])
def load_model_route():
    global _model, _device
    try:
        body  = request.get_json(force=True) or {}
        ckpt  = body.get("ckpt", DEFAULT_CKPT)
        if not Path(ckpt).exists():
            return jsonify({"error": f"Không tìm thấy checkpoint: {ckpt}"}), 404
        dev_str = body.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        _device = torch.device(dev_str)
        with _model_lock:
            _model = load_checkpoint(ckpt, _device)
        n_params = sum(p.numel() for p in _model.parameters())
        return jsonify({
            "status":  "ok",
            "device":  str(_device),
            "ckpt":    ckpt,
            "params":  n_params,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    """
    Nhận một HOẶC nhiều file CSV (field name: 'files' hoặc 'file').
    Với mỗi file: parse → inference (nếu model đã load).
    Trả về list kết quả.
    """
    try:
        # Hỗ trợ cả 'files' (nhiều) và 'file' (một)
        uploaded = request.files.getlist("files") or request.files.getlist("file")
        if not uploaded:
            return jsonify({"error": "Không tìm thấy file nào trong request"}), 400

        with _model_lock:
            model_ready = _model is not None

        results = []
        for f in uploaded:
            fname = (f.filename or "unknown.csv").strip()
            if not fname.lower().endswith(".csv"):
                results.append({"filename": fname, "error": "Chỉ hỗ trợ .csv"})
                continue

            try:
                text = f.read().decode("utf-8-sig", errors="replace")
            except Exception as e:
                results.append({"filename": fname, "error": f"Đọc file thất bại: {e}"})
                continue

            parsed = parse_csv_to_hist(text)
            if "error" in parsed:
                results.append({"filename": fname, "error": parsed["error"]})
                continue

            hist_norm_list = parsed.pop("_hist_norm")
            anchor         = parsed["anchor"]

            if model_ready:
                infer = run_inference(hist_norm_list, anchor["xy"], anchor["heading"])
                if "error" not in infer:
                    parsed["agents"]["AV"]["predictions"] = infer["predictions"]
                    parsed["model_inferred"] = True
                else:
                    parsed["model_inferred"] = False
                    parsed["infer_error"]    = infer["error"]
            else:
                parsed["model_inferred"] = False

            parsed["filename"] = fname
            results.append(parsed)

        return jsonify({"results": results, "count": len(results)})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Nhận hist_norm + anchor, trả về predictions.
    Dùng khi frontend muốn gọi lại inference riêng.
    """
    try:
        body      = request.get_json(force=True) or {}
        hist_norm = body.get("hist_norm")
        anchor    = body.get("anchor", {"xy": [0, 0], "heading": 0})
        if not hist_norm:
            return jsonify({"error": "Thiếu hist_norm"}), 400
        result = run_inference(hist_norm, anchor["xy"], anchor["heading"])
        if "error" in result:
            return jsonify(result), 503
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# NGROK + STARTUP
# ═══════════════════════════════════════════════════════════════════════════
def _start_ngrok(port: int, delay: float = 2.0):
    time.sleep(delay)
    try:
        if NGROK_AUTH_TOKEN:
            conf.get_default().auth_token = NGROK_AUTH_TOKEN
        ngrok.kill()
        tunnel = ngrok.connect(port, "http")
        url    = tunnel.public_url
        print(f"\n{'═' * 60}")
        print(f"  🌐  MotionLM Backend (ngrok): {url}")
        print(f"  ➜   Dán URL này vào Frontend → Backend URL")
        print(f"{'═' * 60}\n")
    except Exception as e:
        print(f"[ngrok] Lỗi: {e}")


if __name__ == "__main__":
    # Auto-load model nếu checkpoint tồn tại
    if DEFAULT_CKPT and Path(DEFAULT_CKPT).is_file():
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[startup] Loading model từ {DEFAULT_CKPT} trên {_device}...")
        _model = load_checkpoint(DEFAULT_CKPT, _device)
        n_p    = sum(p.numel() for p in _model.parameters())
        print(f"[startup] ✓ Model loaded  params={n_p:,}  device={_device}")
    else:
        print(f"[startup] ⚠  Không tìm thấy checkpoint tại {DEFAULT_CKPT}. Model chưa load.")

    threading.Thread(target=_start_ngrok, args=(PORT,), daemon=True).start()
    print(f"🚀  MotionLM Backend  →  http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
