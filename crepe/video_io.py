"""Video tensor <-> frame helpers (same conventions as DiffSynth-Studio)."""

from typing import List

import imageio
import numpy as np
import torch
from einops import reduce
from PIL import Image


def vae_output_to_video(vae_output: torch.Tensor, min_value: float = -1, max_value: float = 1) -> List[Image.Image]:
    """Convert a decoded ``(B, C, T, H, W)`` tensor in [-1, 1] into a list of PIL frames."""
    vae_output = reduce(vae_output, "B C T H W -> T H W C", reduction="mean")
    frames = []
    for image in vae_output:
        image = ((image - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        frames.append(Image.fromarray(image.numpy()))
    return frames


def save_video(frames, save_path, fps: int = 16, quality: int = 8, ffmpeg_params=None):
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def read_video(path) -> torch.Tensor:
    """All frames of a video file as float32 ``(C, T, H, W)`` in [-1, 1].

    Uses decord when available (as for the released training caches), imageio otherwise.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    except ImportError:
        reader = imageio.get_reader(str(path))
        frames = np.stack([np.asarray(f) for f in reader])
        reader.close()
    video = frames.astype(np.float32) / 255.0 * 2 - 1
    return torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()
