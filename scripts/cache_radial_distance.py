#!/usr/bin/env python
"""Cache UniK3D camera-centre distances of PanShot clips at latent resolution.

For each clip, the video frames kept by the Wan VAE (every 4th frame, 21 frames
for 81) are passed to UniK3D together with the clip's exact UCM rays; the
predicted ``distance`` (range along the ray, *not* z-depth) and ``confidence``
are resized to the latent grid (60 x 104) and stored with a validity mask::

    <output_dir>/<video_id>.pth = {"distance": (21, 60, 104), "confidence": ..., "valid": ..., "frame_indices": ...}

Requires UniK3D (https://github.com/lpiccinelli-eth/UniK3D) to be importable.

    python scripts/cache_radial_distance.py --data_root data/UCPE --split train [--num_shards 8 --shard_id 0]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crepe.geometry import ucm_ray_grid  # noqa: E402
from crepe.train.data import load_panshot_metas  # noqa: E402
from crepe.train.radial_supervision import valid_distance_mask  # noqa: E402
from crepe.video_io import read_video  # noqa: E402


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


@torch.inference_mode()
def infer_distance(model, video, x_fov, xi, latent_shape, chunk_size=8):
    latent_frames, latent_h, latent_w = latent_shape
    frames = video.shape[1]
    if frames == 4 * (latent_frames - 1) + 1:
        idx = torch.arange(0, frames, 4)[:latent_frames]
    else:
        idx = torch.linspace(0, frames - 1, latent_frames).round().long()
    video = video.index_select(1, idx).to("cuda")
    _, _, height, width = video.shape
    device = video.device
    rgb = ((rearrange(video, "c t h w -> t c h w") + 1.0) * 127.5).clamp(0.0, 255.0)

    rays = ucm_ray_grid(x_fov=torch.tensor([x_fov], device=device).float().expand(latent_frames).contiguous(),
                        xi=torch.tensor([xi], device=device).float().expand(latent_frames).contiguous(),
                        height=height, width=width, device=device, dtype=torch.float32)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rays = rearrange(rays, "b h w c -> b c h w").contiguous()
    # Pixels outside the UCM domain have NaN rays: feed a dummy ray, mask them afterwards.
    ray_valid = torch.isfinite(rays).all(dim=1)
    rays = torch.where(ray_valid.unsqueeze(1), rays, rays.new_tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1))

    # UniK3D's `infer` is wrapped in an fp16 autocast that is unstable here; run it in fp32.
    raw_infer = _unwrap(model.infer)
    distance, confidence = [], []
    for rgb_c, rays_c in zip(rgb.split(chunk_size), rays.split(chunk_size)):
        with torch.autocast(device_type="cuda", enabled=False):
            out = raw_infer(model, rgb=rgb_c, camera=None, rays=rays_c, normalize=True)
        distance.append(out["distance"].float())
        confidence.append(out["confidence"].float())
    distance, confidence = torch.cat(distance), torch.cat(confidence)
    invalid = (~ray_valid).unsqueeze(1)
    if tuple(distance.shape[-2:]) == invalid.shape[-2:]:
        distance = distance.masked_fill(invalid, float("nan"))
        confidence = confidence.masked_fill(invalid, 0.0)
    if tuple(distance.shape[-2:]) != (latent_h, latent_w):
        distance = F.interpolate(distance, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        confidence = F.interpolate(confidence, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    distance = distance.reshape(latent_frames, latent_h, latent_w)
    confidence = confidence.reshape(latent_frames, latent_h, latent_w)
    return distance, confidence, valid_distance_mask(distance), idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path("data/UCPE"))
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--model_id", default="lpiccinelli/unik3d-vitl")
    p.add_argument("--latent_cache_dir", type=Path, default=None, help="only clips with a latent cache are processed")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=8)
    args = p.parse_args()

    from unik3d.models import UniK3D

    out = args.output_dir or args.data_root / "PanShot" / f"radial-cache-unik3d-vitl-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    latent_dir = args.latent_cache_dir or args.data_root / "PanShot" / f"cache-Wan2.1-T2V-1.3B-{args.split}"
    cached = {p.stem for p in latent_dir.glob("*.pth")}
    metas = [m for m in load_panshot_metas(args.data_root, args.split) if m["video_id"] in cached]
    metas = [m for m in metas[args.shard_id::args.num_shards] if not (out / f"{m['video_id']}.pth").exists()]
    print(f"{len(metas)} clips to process -> {out}")

    model = UniK3D.from_pretrained(args.model_id)
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval().requires_grad_(False)
    for meta in tqdm(metas):
        video = read_video(args.data_root / "PanShot" / f"videos-{args.split}" / f"{meta['video_id']}.mp4")
        latent_shape = ((video.shape[1] - 1) // 4 + 1, video.shape[2] // 8, video.shape[3] // 8)
        distance, confidence, valid, idx = infer_distance(model, video, meta["x_fov"], float(meta["xi"]),
                                                          latent_shape, chunk_size=args.chunk_size)
        record = {"depth_model_id": args.model_id, "latent_shape": latent_shape, "distance": distance.cpu(),
                  "confidence": confidence.cpu(), "valid": valid.cpu(), "frame_indices": idx.cpu()}
        tmp = out / f"{meta['video_id']}.pth.tmp"
        torch.save(record, tmp)
        tmp.replace(out / f"{meta['video_id']}.pth")


if __name__ == "__main__":
    main()
