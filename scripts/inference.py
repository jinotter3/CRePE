#!/usr/bin/env python
"""Generate camera-controlled videos with CRePE.

Batch mode (a JSON list of samples)::

    python scripts/inference.py --checkpoint checkpoints/crepe.safetensors \
        --input demo/lens_showcase.json --output outputs/lens_showcase --seeds 0 1 2

Single sample::

    python scripts/inference.py --checkpoint checkpoints/crepe.safetensors \
        --trajectory demo/trajectories/panshot_aQVLCO-OOqM-11-1-no_rot_aug_0.npy \
        --x_fov 160 --xi 1.5 --prompt "A majestic red fox ..." --output outputs/fox

Each JSON sample has ``caption``, ``trajectory`` (see ``crepe/trajectory.py``),
``x_fov`` / ``xi`` (constants) and optionally ``x_fov_schedule`` /
``xi_schedule`` (``{"type": "linear" | "cosine" | "constant", "start": ..., "end": ...}``)
and a ``name`` used for the output folder.

Camera + radial-map conditioned generation (CRePE-Mix checkpoint) uses the
``sample.json`` written by ``scripts/prepare_video.py``: ``camera`` (npz with pose,
x_fov, xi), ``radial_map`` (UniK3D distances), ``video`` (source clip) and
``denoising_strength`` (< 1 starts from the noised source video: motion transfer)::

    python scripts/inference.py --checkpoint checkpoints/crepe_mix.safetensors \
        --input samples/my_clip/sample.json --output outputs/my_clip
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from crepe import CRePEPipeline, load_trajectory, resolve_lens_value, save_video  # noqa: E402
from crepe.radial_map import RadialMapConfig  # noqa: E402
from crepe.trajectory import lens_label, normalize_panshot_pose  # noqa: E402
from crepe.video_io import read_video  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="CRePE adapter (.safetensors)")
    p.add_argument("--wan_dir", default="models/Wan-AI/Wan2.1-T2V-1.3B", help="Wan2.1-T2V-1.3B folder (downloaded if missing)")
    p.add_argument("--input", help="JSON list of samples")
    p.add_argument("--only", nargs="*", default=None, help="only run samples with these names")
    p.add_argument("--trajectory", help="single-sample trajectory file")
    p.add_argument("--prompt", help="single-sample prompt")
    p.add_argument("--x_fov", type=float, default=None, help="single-sample horizontal FoV (deg)")
    p.add_argument("--xi", type=float, default=None, help="single-sample UCM xi")
    p.add_argument("--output", default="outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_radial_map", action="store_true", help="ignore radial maps (camera-only ablation)")
    p.add_argument("--denoising_strength", type=float, default=None, help="override the samples' value (1 = from noise)")
    p.add_argument("--radial_sigma_mode", default="constant", choices=["constant", "predicted"])
    p.add_argument("--radial_mu_calibration", default="none",
                   choices=["none", "match_predicted_mean", "match_predicted_mean_std"])
    p.add_argument("--radial_teacher_sigma", type=float, default=0.1)
    return p.parse_args()


def load_samples(args):
    if args.input:
        samples = json.load(open(args.input))
        if isinstance(samples, dict):
            samples = [samples]
        for i, s in enumerate(samples):
            s.setdefault("name", f"{i:03d}")
        if args.only:
            samples = [s for s in samples if s["name"] in set(args.only)]
        return samples
    if not (args.trajectory and args.prompt and args.x_fov is not None and args.xi is not None):
        raise SystemExit("Give --input, or --trajectory/--prompt/--x_fov/--xi for a single sample.")
    return [{"name": "sample", "caption": args.prompt, "trajectory": args.trajectory, "x_fov": args.x_fov, "xi": args.xi}]


def main():
    args = parse_args()
    # Same global numerics as the runs that produced the released results
    # (TF32 for float32 matmuls, cuDNN autotuning).
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    samples = load_samples(args)
    pipe = CRePEPipeline.from_pretrained(args.checkpoint, wan_dir=args.wan_dir, device=args.device)
    out_root = Path(args.output)
    radial_cfg = RadialMapConfig(sigma_mode=args.radial_sigma_mode, mu_calibration=args.radial_mu_calibration,
                                 teacher_sigma=args.radial_teacher_sigma)
    for sample in samples:
        if "camera" in sample:   # external video prepared with scripts/prepare_video.py
            cam = np.load(sample["camera"])
            pose = normalize_panshot_pose(cam["pose"].copy(), float(sample.get("radial_scale", 1.0)), zero_first_yaw=True)
            x_fov, xi = float(cam["x_fov"]), float(cam["xi"])
            sample.setdefault("x_fov", x_fov)
            sample.setdefault("xi", xi)
        else:
            pose = load_trajectory(sample["trajectory"], num_frames=args.num_frames)
            x_fov = resolve_lens_value(sample["x_fov"], sample.get("x_fov_schedule"), len(pose))
            xi = resolve_lens_value(sample["xi"], sample.get("xi_schedule"), len(pose))
        radial_map = None
        if sample.get("radial_map") and not args.no_radial_map:
            r = torch.load(sample["radial_map"], map_location="cpu")
            radial_map = {"distance": r["distance"], "valid": r.get("valid"), "scale": float(sample.get("radial_scale", 1.0))}
        strength = args.denoising_strength if args.denoising_strength is not None else float(sample.get("denoising_strength", 1.0))
        input_video = read_video(sample["video"]) if strength < 1.0 else None
        out_dir = out_root / sample["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(sample, fov_label=lens_label(sample["x_fov"], sample.get("x_fov_schedule"), "fov"),
                    xi_label=lens_label(sample["xi"], sample.get("xi_schedule"), "xi"),
                    checkpoint=str(args.checkpoint), steps=args.steps, cfg_scale=args.cfg_scale,
                    denoising_strength=strength, radial_map_used=radial_map is not None,
                    radial_config=vars(radial_cfg) if radial_map is not None else None)
        (out_dir / "sample.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        for seed in args.seeds:
            path = out_dir / f"seed{seed}.mp4"
            if path.exists() and not args.overwrite:
                print(f"skip {path}")
                continue
            frames = pipe(sample["caption"], pose=pose, x_fov=x_fov, xi=xi, seed=seed,
                          num_inference_steps=args.steps, cfg_scale=args.cfg_scale, num_frames=args.num_frames,
                          radial_map=radial_map, radial_config=radial_cfg, input_video=input_video,
                          denoising_strength=strength)
            save_video(frames, path, fps=args.fps, quality=8)
            print(f"saved {path}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
