import argparse
import io
import os
import tarfile
import time
import numpy as np
from typing import Optional, Tuple


# ========================= CONSTANTS =========================
T_HIST   = 20
T_FUTURE = 30
N_ACTORS = 3                    # 0: AV, 1: AGENT, 2: NEAREST OTHER
F_DIM    = 7                    # x, y, vx, vy, speed, heading, type_id

# Đường dẫn mặc định
TRAIN_TAR   = "........"
TRAIN_CACHE = "........"


# ======================= CSV READER =======================
def read_csv(raw_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.genfromtxt(
            io.BytesIO(raw_bytes),
            delimiter=",",
            dtype=None,
            names=True,
            encoding="utf-8",
            invalid_raise=False
        )
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr
    except:
        return None


# ======================= HELPERS =======================
def interp_traj(xy: np.ndarray, target_len: int) -> np.ndarray:
    n = len(xy)
    if n == target_len:
        return xy.astype(np.float32)

    t = np.linspace(0, n-1, target_len)
    s = np.arange(n)
    x = np.interp(t, s, xy[:, 0])
    y = np.interp(t, s, xy[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def get_kinematics(xy: np.ndarray) -> np.ndarray:
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    speed = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    return np.column_stack([xy[:, 0], xy[:, 1], dx, dy, speed, heading])


def ego_normalize(hist: np.ndarray, fut: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rx = hist[0, -1, 0]
    ry = hist[0, -1, 1]
    rh = hist[0, -1, 5]

    c = np.cos(-rh)
    s = np.sin(-rh)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    h = hist.copy()
    f = fut.copy()

    h[:, :, :2] -= [rx, ry]
    f[:, :, :2] -= [rx, ry]

    h[:, :, :2] = h[:, :, :2] @ R.T
    h[:, :, 2:4] = h[:, :, 2:4] @ R.T
    h[:, :, 5] -= rh
    f[:, :, :2] = f[:, :, :2] @ R.T

    return h, f


def get_trajectory(raw: np.ndarray, mask: np.ndarray, total_frames: int) -> Optional[np.ndarray]:
    sub = raw[mask]
    if len(sub) < 2:
        return None

    idx = np.argsort(sub["TIMESTAMP"])
    xy = np.stack([
        sub["X"][idx].astype(np.float32),
        sub["Y"][idx].astype(np.float32)
    ], axis=1)

    return interp_traj(xy, total_frames)


# ======================= PROCESS ONE SCENE =======================
def process_scene(raw_bytes: bytes) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    raw = read_csv(raw_bytes)
    if raw is None:
        return None

    track_ids = raw["TRACK_ID"]
    unique_ids = np.unique(track_ids)

    if len(unique_ids) < 2:
        return None

    av_id = unique_ids[0]
    agent_id = unique_ids[1]

    av_xy = get_trajectory(raw, track_ids == av_id, T_HIST + T_FUTURE)
    agent_xy = get_trajectory(raw, track_ids == agent_id, T_HIST + T_FUTURE)

    if av_xy is None or agent_xy is None:
        return None

    # Tìm nearest other
    other_xy = None
    av_hist_pos = av_xy[:T_HIST, :2]

    other_mask = (track_ids != av_id) & (track_ids != agent_id)

    if np.any(other_mask):
        other_raw = raw[other_mask]
        other_track_ids = other_raw["TRACK_ID"]
        other_unique = np.unique(other_track_ids)

        min_dist = float('inf')
        best_other = None

        for tid in other_unique:
            mask = other_track_ids == tid
            if np.sum(mask) < 2:
                continue
            traj = get_trajectory(other_raw, mask, T_HIST + T_FUTURE)
            if traj is None:
                continue

            dist = np.mean(np.linalg.norm(traj[:T_HIST, :2] - av_hist_pos, axis=1))
            if dist < min_dist:
                min_dist = dist
                best_other = traj

        if best_other is not None:
            other_xy = best_other

    if other_xy is None:
        other_xy = av_xy.copy()

    # Build features
    all_xy = np.stack([av_xy, agent_xy, other_xy], axis=0)

    kin = np.stack([get_kinematics(all_xy[i]) for i in range(N_ACTORS)])

    hist = np.zeros((N_ACTORS, T_HIST, F_DIM), dtype=np.float32)
    fut = np.zeros((N_ACTORS, T_FUTURE, 2), dtype=np.float32)

    hist[:, :, :6] = kin[:, :T_HIST, :]
    hist[:, :, 6] = np.array([0, 1, 2])[:, None]
    fut = all_xy[:, T_HIST:, :2].copy()

    hist, fut = ego_normalize(hist, fut)

    return hist.astype(np.float16), fut.astype(np.float16)


# ======================= MAIN BUILD =======================
def build_cache(tar_path: str, cache_dir: str, force: bool = False):
    hist_path = os.path.join(cache_dir, "hist.npy")
    fut_path = os.path.join(cache_dir, "fut.npy")

    if not force and os.path.exists(hist_path) and os.path.exists(fut_path):
        n = np.load(hist_path, mmap_mode='r').shape[0]
        print(f"✅ Cache đã tồn tại ({n:,} scenes). Dùng --force để rebuild.")
        return

    os.makedirs(cache_dir, exist_ok=True)

    print("🔄 Đang bắt đầu xử lý tar.gz...")

    hists = []
    futs = []
    n_ok = 0
    n_skip = 0
    start_time = time.time()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
        total = len(members)

        for i, member in enumerate(members):
            fobj = tar.extractfile(member)
            if fobj is None:
                n_skip += 1
                continue

            result = process_scene(fobj.read())

            if result is None:
                n_skip += 1
            else:
                hists.append(result[0])
                futs.append(result[1])
                n_ok += 1

            if (i + 1) % 1000 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                print(f"[{i+1:6,}/{total:,}]  OK: {n_ok:5,} | Skip: {n_skip:5,} | "
                      f"Time: {elapsed:.1f}s")

    if n_ok == 0:
        print("❌ Không có scene hợp lệ nào!")
        return

    print("\n💾 Đang lưu cache...")
    hist_array = np.stack(hists).astype(np.float16)
    fut_array = np.stack(futs).astype(np.float16)

    np.save(hist_path, hist_array)
    np.save(fut_path, fut_array)

    elapsed = time.time() - start_time
    print("="*70)
    print("✅ HOÀN TẤT!")
    print(f"   Scenes hợp lệ : {n_ok:,}")
    print(f"   Scenes bỏ qua : {n_skip:,}")
    print(f"   hist shape    : {hist_array.shape}")
    print(f"   fut shape     : {fut_array.shape}")
    print(f"   Thời gian     : {elapsed:.1f} giây")
    print(f"   Lưu tại       : {cache_dir}")
    print("="*70)


# ======================= ARGS (FIXED FOR COLAB) =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=TRAIN_TAR,   help="Đường dẫn file tar.gz")
    parser.add_argument("--output", default=TRAIN_CACHE, help="Thư mục lưu cache")
    parser.add_argument("--force",  action="store_true", help="Buộc rebuild cache")

    # Sửa lỗi Colab bằng parse_known_args()
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"⚠️  Bỏ qua các argument không rõ: {unknown}")

    build_cache(args.input, args.output, args.force)


if __name__ == "__main__":
    main()
