#!/usr/bin/env python
"""Pre-compute Wan2.1 VAE latents and umT5 text contexts for PanShot clips.

Writes ``<output_dir>/<video_id>.pth`` = ``{"input_latents": (16, 21, 60, 104), "context": (512, 4096)}``
(bfloat16), which is what ``crepe.train.data.PanShotTrainDataset`` reads.

    python scripts/cache_latents.py --data_root data/UCPE --split train [--num_shards 8 --shard_id 0]
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crepe.train.data import load_panshot_metas  # noqa: E402
from crepe.video_io import read_video  # noqa: E402
from crepe.wan import WanPrompter, load_text_encoder, load_vae, resolve_wan_dir, tokenizer_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=Path("data/UCPE"))
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--wan_dir", default="models/Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = args.output_dir or args.data_root / "PanShot" / f"cache-Wan2.1-T2V-1.3B-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    metas = load_panshot_metas(args.data_root, args.split)[args.shard_id::args.num_shards]
    metas = [m for m in metas if not (out / f"{m['video_id']}.pth").exists()]
    print(f"{len(metas)} clips to cache -> {out}")

    wan_dir = resolve_wan_dir(args.wan_dir)
    device = torch.device(args.device)
    vae = load_vae(wan_dir, torch.bfloat16, device)
    prompter = WanPrompter(tokenizer_path(wan_dir), text_encoder=load_text_encoder(wan_dir, torch.bfloat16, device))
    for meta in tqdm(metas):
        video = read_video(args.data_root / "PanShot" / f"videos-{args.split}" / f"{meta['video_id']}.mp4")
        with torch.no_grad():
            # bf16 input, as in the original (Lightning bf16-true) caching run.
            latents = vae.encode(video.unsqueeze(0).to(torch.bfloat16), device=device, tiled=False)
            latents = latents.to(dtype=torch.bfloat16, device=device)
            context = prompter.encode_prompt(meta["caption"], device=device)
        tmp = out / f"{meta['video_id']}.pth.tmp"
        torch.save({"input_latents": latents[0].cpu(), "context": context[0].cpu()}, tmp)
        tmp.replace(out / f"{meta['video_id']}.pth")


if __name__ == "__main__":
    main()
