"""Camera trajectories and lens schedules.

A trajectory is a sequence of camera-to-world matrices ``(T, 3, 4)`` in the
OpenCV convention (x right, y down, z forward) with the world ``-y`` axis
pointing up, and translations in the scene-normalised units used for training.

Supported files:
    * ``.npy``  - an already-normalised ``(T, 3, 4)`` or ``(T, 4, 4)`` array
                  (all trajectories under ``demo/trajectories`` are of this kind);
    * ``.txt``  - a RealEstate10K camera file, normalised like the paper's
                  evaluation (relative to the first frame, scaled by
                  ``RE10K_TRAJ_SCALE``).
Raw PanShot poses can be normalised with :func:`normalize_panshot_pose`.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from einops import repeat

# Global translation scale applied to RealEstate10K trajectories
# (``traj_normalization.txt`` of the UCPE evaluation protocol).
RE10K_TRAJ_SCALE = 0.9569655060768127441


def normalize_panshot_pose(pose: np.ndarray, near_depth: float, zero_first_yaw: bool = True) -> np.ndarray:
    """Normalise a raw PanShot ``(T, 3, 4)`` c2w trajectory (in place, like training).

    Translations are divided by the clip's near-plane depth. With
    ``zero_first_yaw`` the world is rotated about its vertical axis so the first
    camera looks along the y-z plane; otherwise the first camera becomes the
    world frame.
    """
    pose[..., 3] /= near_depth
    if zero_first_yaw:
        forward = pose[0, :, 2]
        forward_xy = np.array([forward[0], 0, forward[2]])
        forward_xy /= np.linalg.norm(forward_xy) + 1e-8
        theta = np.arctan2(forward_xy[0], forward_xy[2])
        c, s = np.cos(-theta), np.sin(-theta)
        R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=pose.dtype)
        pose[..., :3] = R_y[None] @ pose[..., :3]
        pose[..., 3] = (R_y[None] @ pose[..., 3:4]).squeeze(-1)
        return pose
    last_row = repeat(np.array([0, 0, 0, 1], dtype=pose.dtype), "n -> t 1 n", t=pose.shape[0])
    c2w = np.concatenate([pose, last_row], axis=-2)
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    return c2w[:, :3]


def load_re10k_trajectory(path: Union[str, Path], num_frames: int = 81, scale: Optional[float] = RE10K_TRAJ_SCALE) -> np.ndarray:
    """Read a RealEstate10K camera file as a first-frame-relative ``(T, 3, 4)`` c2w array."""
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("http"):
                lines.append(line)
                if len(lines) >= num_frames:
                    break
    poses = np.stack([np.array(list(map(float, ln.split()[7:]))).reshape(3, 4) for ln in lines], axis=0)
    last_row = repeat(np.array([0, 0, 0, 1], dtype=poses.dtype), "n -> t 1 n", t=poses.shape[0])
    c2w = np.linalg.inv(np.concatenate([poses, last_row], axis=-2))
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    poses = c2w[:, :3]
    if scale is not None:
        poses[..., 3] *= scale
    return poses


def load_trajectory(path: Union[str, Path], num_frames: int = 81) -> np.ndarray:
    """Load a trajectory file (see module docstring). Keeps the stored dtype."""
    path = Path(path)
    if path.suffix == ".npy":
        pose = np.load(path)
        if pose.ndim != 3 or pose.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError(f"{path}: expected (T, 3, 4) or (T, 4, 4), got {pose.shape}")
        return pose[:, :3, :4]
    if path.suffix == ".txt":
        return load_re10k_trajectory(path, num_frames=num_frames)
    raise ValueError(f"Unsupported trajectory file: {path}")


def resolve_lens_value(value, schedule: Optional[dict], num_frames: int):
    """Per-sample lens parameter: a constant (float) or a per-frame float32 array.

    ``schedule`` is ``{"type": "constant" | "linear" | "cosine", "start": a, "end": b}``.
    """
    if schedule is None:
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"lens value list must be 1D, got {arr.shape}")
            if arr.shape[0] == num_frames:
                return arr
            return arr[np.linspace(0, arr.shape[0] - 1, num_frames).round().astype(np.int64)]
        return value
    kind = schedule.get("type", "linear")
    start = float(schedule.get("start", schedule.get("value", value)))
    end = float(schedule.get("end", start))
    if kind == "constant":
        arr = np.full(num_frames, start, dtype=np.float32)
    elif kind == "linear":
        arr = np.linspace(start, end, num_frames, dtype=np.float32)
    elif kind == "cosine":
        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
        arr = start + (end - start) * (0.5 - 0.5 * np.cos(np.pi * t))
    else:
        raise ValueError(f"Unsupported schedule type {kind!r}; expected constant, linear or cosine")
    return arr.astype(np.float32)


def lens_label(value, schedule: Optional[dict], prefix: str) -> str:
    src = schedule if schedule is not None else value
    if isinstance(src, dict):
        start = float(src.get("start", src.get("value", 0.0)))
        end = float(src.get("end", start))
    elif isinstance(src, (list, tuple)):
        start, end = float(src[0]), float(src[-1])
    else:
        start = end = float(src)
    return f"{prefix}{start:.2f}" if abs(start - end) < 1e-6 else f"{prefix}{start:.2f}to{end:.2f}"
