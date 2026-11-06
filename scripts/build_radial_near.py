#!/usr/bin/env python
"""Compute one teacher near scale per PanFlow source clip from cached UniK3D distances.

For every cached view: ``view_near = median_t( quantile_25( valid distances of frame t ) )``
with valid distances in ``[1e-3, 100]``. A source clip's scale is the median
``view_near`` over its views without rotation augmentation (falling back to all
cached views, and then to the remaining augmentations if none of the selected
views has valid pixels). The same scale is written for every augmentation of
the source, so all views of a scene share one metric normalisation.

    python scripts/build_radial_near.py --data_root data/UCPE \
        --radial_cache_dir data/UCPE/PanShot/radial-cache-unik3d-vitl-train \
        --output data/UCPE/PanShot/radial-near-unik3d-vitl-train.jsonl
"""

import argparse
import concurrent.futures
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

RAW_MIN, RAW_MAX = 1e-3, 100.0


def view_near(path: str):
    record = torch.load(path, map_location="cpu", weights_only=True)
    distance = torch.as_tensor(record["distance"]).float().numpy()
    valid = np.isfinite(distance) & (distance >= RAW_MIN) & (distance <= RAW_MAX)
    if record.get("valid") is not None:
        valid &= torch.as_tensor(record["valid"]).numpy().astype(bool)
    flat_d = distance.reshape(distance.shape[0], -1)
    flat_v = valid.reshape(valid.shape[0], -1)
    frame_near = np.full(distance.shape[0], np.nan)
    for t in range(distance.shape[0]):
        values = flat_d[t, flat_v[t]]
        if values.size:
            frame_near[t] = float(np.quantile(values, 0.25))
    finite = frame_near[np.isfinite(frame_near)]
    finite = finite[finite > 0.0]
    return float(np.median(finite)) if finite.size else float("nan")


def ok(v):
    return np.isfinite(v) and v > 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("data/UCPE"))
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--radial_cache_dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    videos_by_source, preferred_by_source, source_by_video = {}, {}, {}
    for meta_file in sorted((args.data_root / "PanShot" / f"meta-{args.split}").glob("*.json")):
        src = meta_file.stem
        videos_by_source[src], preferred_by_source[src] = [], []
        for match in json.load(open(meta_file)):
            for video in match["videos"]:
                vid = str(video["video"])
                source_by_video[vid] = src
                videos_by_source[src].append(vid)
                if not video.get("rot_aug"):
                    preferred_by_source[src].append(vid)
    paths = {v: args.radial_cache_dir / f"{v}.pth" for v in source_by_video if (args.radial_cache_dir / f"{v}.pth").is_file()}
    if not paths:
        raise SystemExit(f"no cached distances in {args.radial_cache_dir}")

    selected = {}
    for src, videos in videos_by_source.items():
        chosen = [v for v in preferred_by_source[src] if v in paths] or [v for v in videos if v in paths]
        if chosen:
            selected[src] = chosen

    def compute(video_ids):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            return dict(zip(video_ids, ex.map(view_near, [str(paths[v]) for v in video_ids], chunksize=8)))

    near_by_video = compute([v for vs in selected.values() for v in vs])
    for src, vs in list(selected.items()):
        if not any(ok(near_by_video[v]) for v in vs):
            rest = [v for v in videos_by_source[src] if v in paths and v not in set(vs)]
            near_by_video.update(compute(rest))
            if not any(ok(near_by_video[v]) for v in rest):
                raise RuntimeError(f"source clip {src} has no valid teacher-distance view")
            selected[src] = rest

    near_by_source = {src: float(np.median([near_by_video[v] for v in vs if ok(near_by_video[v])]))
                      for src, vs in selected.items()}
    records = [{"video_id": v, "source_clip_id": source_by_video[v], "unik_near": near_by_source[source_by_video[v]]}
               for v in sorted(paths)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=args.output.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    print(f"wrote {len(records)} rows ({len(near_by_source)} source clips) to {args.output}")


if __name__ == "__main__":
    main()
