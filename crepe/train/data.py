"""PanShot training data with cached latents and cached UniK3D radial distances.

Expected layout (``data_root``, default ``data/UCPE`` as in UCPE)::

    data/UCPE/
    ├── PanFlow/near_plane_depth-train.jsonl     # shipped with this repo
    └── PanShot/
        ├── captioned-train.jsonl
        ├── meta-train/*.json
        ├── pose-train/*.npy
        ├── videos-train/*.mp4                    # only needed to build the caches
        ├── cache-Wan2.1-T2V-1.3B-train/*.pth     # scripts/cache_latents.py
        ├── radial-cache-unik3d-vitl-train/*.pth  # scripts/cache_radial_distance.py
        └── radial-near-unik3d-vitl-train.jsonl   # scripts/build_radial_near.py
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..trajectory import normalize_panshot_pose


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_near_plane_depths(path) -> dict:
    """``{"<youtube id>-<clip id>": near_depth}`` from PanFlow's near-plane file."""
    return {f"{o['video_id']}-{o['clip_id']}": o["near_depth"] for o in read_jsonl(path)}


def load_radial_near(path) -> dict:
    """``{video_id: unik_near}`` teacher scales written by ``scripts/build_radial_near.py``."""
    out = {}
    for o in read_jsonl(path):
        value = float(o["unik_near"])
        if not np.isfinite(value) or value <= 1e-6:
            raise ValueError(f"invalid unik_near for {o['video_id']}: {value}")
        out[o["video_id"]] = value
    return out


def load_panshot_metas(data_root: Path, split: str) -> list:
    """All captioned PanShot clips of a split with their lens and near-plane depth."""
    near_depths = load_near_plane_depths(data_root / "PanFlow" / f"near_plane_depth-{split}.jsonl")
    metas = {}
    for meta_file in (data_root / "PanShot" / f"meta-{split}").glob("*.json"):
        for match in json.load(open(meta_file)):
            for video in match["videos"]:
                metas[video["video"]] = {
                    "pose_id": video["pose"],
                    "x_fov": float(video["x_fov"]),
                    "xi": video["xi"],
                    "near_depth": near_depths[meta_file.stem],
                    "source_clip_id": meta_file.stem,
                    "rot_aug": bool(video.get("rot_aug")),
                }
    out = []
    for obj in read_jsonl(data_root / "PanShot" / f"captioned-{split}.jsonl"):
        if obj["video"] in metas:
            meta = dict(metas[obj["video"]], caption=obj["caption"], video_id=obj["video"])
            out.append(meta)
    return out


class PanShotTrainDataset(Dataset):
    def __init__(
        self,
        data_root="data/UCPE",
        split: str = "train",
        latent_cache_dir: Optional[str] = None,
        radial_cache_dir: Optional[str] = None,
        radial_near_file: Optional[str] = None,
        zero_first_yaw: bool = True,
        video_ids: Optional[Iterable[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.zero_first_yaw = zero_first_yaw
        self.latent_cache_dir = Path(latent_cache_dir or self.data_root / "PanShot" / f"cache-Wan2.1-T2V-1.3B-{split}")
        self.radial_cache_dir = Path(radial_cache_dir) if radial_cache_dir else None
        self.radial_near = load_radial_near(radial_near_file) if radial_near_file else None

        metas = load_panshot_metas(self.data_root, split)
        cached = {p.stem for p in self.latent_cache_dir.glob("*.pth")}
        metas = [m for m in metas if m["video_id"] in cached]
        if self.radial_cache_dir is not None:
            radial = {p.stem for p in self.radial_cache_dir.glob("*.pth")}
            metas = [m for m in metas if m["video_id"] in radial]
        if video_ids is not None:
            video_ids = set(video_ids)
            metas = [m for m in metas if m["video_id"] in video_ids]
        if self.radial_near is not None:
            missing = [m["video_id"] for m in metas if m["video_id"] not in self.radial_near]
            if missing:
                raise KeyError(f"radial near scale missing for {len(missing)} clips, e.g. {missing[:3]}")
        if not metas:
            raise RuntimeError(f"No training clips found under {self.data_root} (latent cache: {self.latent_cache_dir})")
        self.metas = metas
        print(f"[PanShotTrainDataset] {len(self.metas)} clips")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        data = {k: meta[k] for k in ("video_id", "caption", "x_fov", "xi", "near_depth")}
        cache = torch.load(self.latent_cache_dir / f"{meta['video_id']}.pth", map_location="cpu")
        data["input_latents"] = cache["input_latents"]
        data["context"] = cache["context"]
        if self.radial_cache_dir is not None:
            radial = torch.load(self.radial_cache_dir / f"{meta['video_id']}.pth", map_location="cpu")
            data["radial_distance"] = radial["distance"]
            data["radial_valid"] = radial["valid"]
        if self.radial_near is not None:
            data["radial_near"] = np.float32(self.radial_near[meta["video_id"]])
        pose = np.load(self.data_root / "PanShot" / f"pose-{self.split}" / Path(meta["pose_id"]).with_suffix(".npy"))
        data["pose"] = normalize_panshot_pose(pose, meta["near_depth"], zero_first_yaw=self.zero_first_yaw)
        return data


class PanShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str = "data/UCPE",
        latent_cache_dir: Optional[str] = None,
        radial_cache_dir: Optional[str] = None,
        radial_near_file: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        zero_first_yaw: bool = True,
        train_video_ids_file: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        h = self.hparams
        video_ids = json.load(open(h.train_video_ids_file)) if h.train_video_ids_file else None
        dataset = PanShotTrainDataset(h.data_root, "train", h.latent_cache_dir, h.radial_cache_dir,
                                      h.radial_near_file, h.zero_first_yaw, video_ids)
        return DataLoader(dataset, batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers)
