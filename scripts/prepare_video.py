#!/usr/bin/env python
"""Prepare an external video for camera + radial-map conditioned generation (CRePE-Mix).

    MP4 / frame folder -> 81-frame 832x480 clip -> ViPE camera -> UniK3D radial map

Writes ``<out_dir>/<name>/``::

    video.mp4      standardized clip (also the motion-transfer source)
    camera.npz     pose (81, 3, 4) camera-to-world relative to frame 0, x_fov, xi
    radial.pth     UniK3D distance / confidence / validity at latent resolution (21, 60, 104)
    sample.json    entry for scripts/inference.py (add your prompt with --prompt)

Requirements: ViPE (https://github.com/nv-tlabs/vipe) for the camera, UniK3D
(https://github.com/lpiccinelli-eth/UniK3D) for the radial map.

    python scripts/prepare_video.py --input source.mp4 --name my_clip --prompt "..." \
        --vipe_dir /path/to/vipe [--vipe_python /path/to/vipe/env/python]
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------------- frames
def sample_indices(num_frames, target, mode, start=0, stride=1, source_fps=None, output_fps=None):
    if mode == "first":
        count = min(1 + (num_frames - start - 1) // stride, target)
        idx = start + np.arange(count, dtype=np.int64) * stride
        if len(idx) < target:  # repeat the last frame of short clips
            idx = np.concatenate([idx, np.full(target - len(idx), idx[-1], dtype=np.int64)])
        return idx
    if mode == "fps":
        pos = start + np.arange(target, dtype=np.float64) * (source_fps / output_fps) * stride
        return np.clip(np.rint(pos), start, num_frames - 1).astype(np.int64)
    if mode == "linspace":
        return np.linspace(start, num_frames - 1, target).round().astype(np.int64)
    raise ValueError(mode)


def read_source(args):
    if args.input.is_dir():
        paths = sorted(p for p in args.input.glob(args.frame_glob) if p.is_file())
        idx = sample_indices(len(paths), args.frames, args.frame_sample, args.start, args.stride,
                             args.source_fps, args.fps)
        return np.stack([np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx]), idx
    from decord import VideoReader, cpu
    vr = VideoReader(str(args.input), ctx=cpu(0), num_threads=1)
    source_fps = args.source_fps or float(vr.get_avg_fps())
    idx = sample_indices(len(vr), args.frames, args.frame_sample, args.start, args.stride, source_fps, args.fps)
    return vr.get_batch(idx.tolist()).asnumpy(), idx


def resize(frame, width, height, mode):
    image = Image.fromarray(frame)
    sw, sh = image.size
    if mode == "stretch":
        return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)
    if mode == "crop":
        scale = max(width / sw, height / sh)
        rw, rh = max(width, int(math.ceil(sw * scale))), max(height, int(math.ceil(sh * scale)))
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        left, top = max(0, (rw - width) // 2), max(0, (rh - height) // 2)
        return np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.uint8)
    if mode == "letterbox":
        scale = min(width / sw, height / sh)
        rw, rh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        image = image.resize((rw, rh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        canvas.paste(image, (max(0, (width - rw) // 2), max(0, (height - rh) // 2)))
        return np.asarray(canvas, dtype=np.uint8)
    raise ValueError(mode)


def write_video(frames, path, fps):
    h, w = frames.shape[1:3]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
           "-r", str(fps), "-crf", "16", "-preset", "slow", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


# ----------------------------------------------------------------------------- camera
def run_vipe(args, video_dir: Path, vipe_out: Path):
    cmd = [args.vipe_python, str(Path(args.vipe_dir) / "run.py"), "pipeline=default", "streams=raw_mp4_stream",
           f"streams.base_path={video_dir}", f"pipeline.output.path={vipe_out}",
           "pipeline.output.save_artifacts=true", "pipeline.post.depth_align_model=null"]
    print("[prepare] running ViPE:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.vipe_dir)


def load_vipe_camera(vipe_out: Path, name: str, frames: int, width: int, pose_scale: float, fallback_fov: float):
    with np.load(vipe_out / "pose" / f"{name}.npz") as data:
        c2w = data["data"].astype(np.float32)
    if c2w.shape[0] != frames:
        c2w = c2w[np.linspace(0, c2w.shape[0] - 1, frames).round().astype(np.int64)]
    c2w = np.linalg.inv(c2w[0])[None].astype(np.float32) @ c2w
    c2w[:, :3, 3] *= float(pose_scale)
    x_fov = float(fallback_fov)
    intr_path = vipe_out / "intrinsics" / f"{name}.npz"
    if intr_path.exists():
        with np.load(intr_path) as data:
            intr = data["data"].astype(np.float32)
        fx = np.nanmedian(intr[:, 0]) if intr.ndim == 2 and intr.shape[1] >= 1 else float("nan")
        if np.isfinite(fx) and fx > 0:
            x_fov = float(math.degrees(2.0 * math.atan(float(width) / (2.0 * float(fx)))))
    return c2w[:, :3, :4].astype(np.float32), x_fov


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="video file or folder of frames")
    p.add_argument("--name", required=True)
    p.add_argument("--out_dir", type=Path, default=Path("samples"))
    p.add_argument("--prompt", default="")
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frame_sample", choices=["first", "linspace", "fps"], default="first")
    p.add_argument("--source_fps", type=float, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frame_glob", default="*.jpg")
    p.add_argument("--resize", choices=["letterbox", "crop", "stretch"], default="letterbox")
    p.add_argument("--vipe_dir", default=None, help="ViPE repository (contains run.py)")
    p.add_argument("--vipe_python", default=sys.executable)
    p.add_argument("--vipe_out", type=Path, default=None, help="reuse an existing ViPE output folder")
    p.add_argument("--x_fov", type=float, default=None, help="override the FoV estimated by ViPE (degrees)")
    p.add_argument("--fallback_x_fov", type=float, default=90.0)
    p.add_argument("--xi", type=float, default=0.0, help="UCM xi of the source camera (0 = pinhole)")
    p.add_argument("--pose_scale", type=float, default=1.0)
    p.add_argument("--pose_npy", type=Path, default=None,
                   help="known (T, 3, 4) camera-to-world poses (e.g. PanShot pose-*/<clip>.npy); skips ViPE")
    p.add_argument("--near_depth", type=float, default=1.0,
                   help="scene scale: poses and radial distances are divided by it (PanFlow near-plane depth for PanShot)")
    p.add_argument("--unik3d_model", default="lpiccinelli/unik3d-vitl")
    p.add_argument("--skip_radial", action="store_true")
    args = p.parse_args()

    out = args.out_dir / args.name
    (out / "clip").mkdir(parents=True, exist_ok=True)
    video_path = out / "clip" / f"{args.name}.mp4"   # ViPE names its outputs after the file
    if not video_path.exists() and args.input.suffix == ".mp4":
        # Already in the model's format (e.g. PanShot clips): copy instead of re-encoding.
        import imageio
        reader = imageio.get_reader(str(args.input))
        meta, n = reader.get_meta_data(), reader.count_frames()
        reader.close()
        if n == args.frames and tuple(meta.get("size", ())) == (args.width, args.height):
            import shutil
            shutil.copyfile(args.input, video_path)
    if not video_path.exists():
        frames, idx = read_source(args)
        frames = np.stack([resize(f, args.width, args.height, args.resize) for f in frames])
        write_video(frames, video_path, args.fps)
        json.dump({"source": str(args.input), "frame_indices": idx.tolist(), "resize": args.resize, "fps": args.fps},
                  open(out / "standardize.json", "w"), indent=1)

    if args.pose_npy is not None:
        if args.x_fov is None:
            raise SystemExit("--pose_npy needs --x_fov (and --xi for non-pinhole cameras)")
        pose, x_fov = np.load(args.pose_npy)[:, :3, :4], float(args.x_fov)
    else:
        vipe_out = args.vipe_out or out / "vipe"
        if not (vipe_out / "pose" / f"{args.name}.npz").exists():
            if args.vipe_dir is None:
                raise SystemExit("No ViPE output found; pass --vipe_dir (or --vipe_out with existing results).")
            run_vipe(args, video_path.parent.resolve(), vipe_out.resolve())
        pose, x_fov = load_vipe_camera(vipe_out, args.name, args.frames, args.width, args.pose_scale, args.fallback_x_fov)
    if args.x_fov is not None:
        x_fov = float(args.x_fov)
    np.savez(out / "camera.npz", pose=pose, x_fov=np.float64(x_fov), xi=np.float64(args.xi))
    print(f"[prepare] camera: x_fov={x_fov:.2f} deg, xi={args.xi}")

    if not args.skip_radial and not (out / "radial.pth").exists():
        from unik3d.models import UniK3D
        from crepe.video_io import read_video
        from scripts.cache_radial_distance import infer_distance
        model = UniK3D.from_pretrained(args.unik3d_model)
        model.resolution_level = 0
        model.interpolation_mode = "bilinear"
        model = model.to("cuda").eval().requires_grad_(False)
        video = read_video(video_path)
        shape = ((video.shape[1] - 1) // 4 + 1, args.height // 8, args.width // 8)
        distance, confidence, valid, idx = infer_distance(model, video, x_fov, float(args.xi), shape)
        torch.save({"depth_model_id": args.unik3d_model, "distance": distance.cpu(), "confidence": confidence.cpu(),
                    "valid": valid.cpu(), "frame_indices": idx.cpu()}, out / "radial.pth")

    sample = {"name": args.name, "caption": args.prompt, "video": str(video_path), "camera": str(out / "camera.npz"),
              "radial_map": str(out / "radial.pth"), "radial_scale": float(args.near_depth), "denoising_strength": 1.0}
    json.dump(sample, open(out / "sample.json", "w"), indent=1)
    print(f"[prepare] wrote {out / 'sample.json'}")


if __name__ == "__main__":
    main()
