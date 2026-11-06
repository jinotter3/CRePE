"""Camera-controlled text-to-video generation with CRePE.

Example::

    import numpy as np
    from crepe import CRePEPipeline, load_trajectory

    pipe = CRePEPipeline.from_pretrained("checkpoints/crepe.safetensors", device="cuda")
    pose = load_trajectory("demo/trajectories/panshot_aQVLCO-OOqM-11-1-no_rot_aug_0.npy")
    frames = pipe("A red fox in an autumn forest ...", pose=pose, x_fov=160.0, xi=1.5, seed=0)
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from .adapter import CRePEConfig, attach_camera_adapters
from .camera_inputs import PATCH_FACTOR, as_batched_tensors, build_camera_context
from .checkpoint import load_adapter_file, load_adapter_weights
from .radial_map import RadialMapConfig, build_radial_condition
from .video_io import vae_output_to_video
from .wan import FlowMatchScheduler, WanPrompter, load_dit, load_text_encoder, load_vae, resolve_wan_dir, tokenizer_path

# Wan2.1's default negative prompt (used for all released results).
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


class CRePEPipeline:
    def __init__(self, dit, vae, text_encoder, tokenizer_dir, config: CRePEConfig,
                 device: Union[str, torch.device] = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.prompter = WanPrompter(tokenizer_dir, text_encoder=text_encoder)
        self.config = config
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

    @classmethod
    def from_pretrained(
        cls,
        adapter_path: Union[str, Path],
        wan_dir: Union[str, Path, None] = None,
        device: Union[str, torch.device] = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "CRePEPipeline":
        """Load Wan2.1-T2V-1.3B from ``wan_dir`` (downloaded if missing) plus a CRePE adapter."""
        wan_dir = resolve_wan_dir(wan_dir)
        state_dict, config, _ = load_adapter_file(adapter_path)
        dit = load_dit(wan_dir, torch_dtype=torch_dtype, device="cpu")
        attach_camera_adapters(dit, config)
        dit = dit.to(dtype=torch_dtype)
        load_adapter_weights(dit, state_dict)
        dit = dit.to(device).eval().requires_grad_(False)
        vae = load_vae(wan_dir, torch_dtype=torch_dtype, device=device).requires_grad_(False)
        text_encoder = load_text_encoder(wan_dir, torch_dtype=torch_dtype, device=device).requires_grad_(False)
        return cls(dit, vae, text_encoder, tokenizer_path(wan_dir), config, device=device, torch_dtype=torch_dtype)

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.prompter.encode_prompt(prompt, device=self.device)

    def camera_context(self, pose, x_fov, xi, camera_dtype=torch.bfloat16) -> dict:
        """Camera context for one trajectory (numpy / float inputs) or a batch (tensors with a batch dim)."""
        if not torch.is_tensor(pose):
            pose, x_fov, xi = as_batched_tensors(pose, x_fov, xi)
        return build_camera_context(pose, x_fov, xi, height=self.config.height, width=self.config.width,
                                    device=self.device, camera_dtype=camera_dtype,
                                    ray_offsets=self.config.ray_offsets)

    def generate_noise(self, num_frames: int, seed: Optional[int], height: int, width: int) -> torch.Tensor:
        shape = (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
        generator = None if seed is None else torch.Generator("cpu").manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
        return noise.to(dtype=self.torch_dtype, device=self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        pose: Union[np.ndarray, torch.Tensor],
        x_fov: Union[float, np.ndarray, torch.Tensor],
        xi: Union[float, np.ndarray, torch.Tensor],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: Optional[int] = 0,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        sigma_shift: float = 5.0,
        num_frames: int = 81,
        tiled: bool = False,
        camera_dtype: Optional[torch.dtype] = torch.bfloat16,
        radial_map: Optional[dict] = None,
        radial_config: Optional[RadialMapConfig] = None,
        input_video: Optional[torch.Tensor] = None,
        denoising_strength: float = 1.0,
        progress: bool = True,
    ) -> List[Image.Image]:
        """Generate ``num_frames`` frames that follow ``pose`` under the UCM lens ``(x_fov, xi)``.

        Args:
            prompt: text prompt.
            pose: ``(num_frames, 3, 4)`` camera-to-world trajectory (see ``crepe.trajectory``).
            x_fov: horizontal FoV in degrees (scalar or per-frame array).
            xi: UCM ``xi`` (scalar or per-frame array; ``0`` = pinhole).
            radial_map: optional external radial map for a CRePE-Mix checkpoint:
                ``{"distance": (T, H, W), "valid": (T, H, W) or None, "scale": float}``
                (see :mod:`crepe.radial_map`), applied with ``radial_config``.
            input_video: optional ``(C, T, H, W)`` video in [-1, 1]; with
                ``denoising_strength < 1`` generation starts from this video noised to
                that strength (source-video motion transfer).
        """
        height, width = self.config.height, self.config.width
        if pose.shape[-3] != num_frames:
            raise ValueError(f"pose has {pose.shape[-3]} frames, expected {num_frames}")
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        context_posi = self.encode_prompt(prompt)
        context_nega = self.encode_prompt(negative_prompt) if cfg_scale != 1.0 else None
        camera_ctx = self.camera_context(pose, x_fov, xi, camera_dtype=camera_dtype)
        if radial_map is not None:
            condition = build_radial_condition(
                radial_map["distance"], radial_map.get("valid"), radial_map.get("scale", 1.0),
                num_cameras=camera_ctx["w2cs"].shape[1], patches_y=height // PATCH_FACTOR,
                patches_x=width // PATCH_FACTOR, device=self.device)
            camera_ctx["radial_map"] = dict(condition, config=radial_config or RadialMapConfig())
        latents = self.generate_noise(num_frames, seed, height, width)
        if input_video is not None:
            video = input_video.unsqueeze(0) if input_video.ndim == 4 else input_video
            video = video.to(dtype=camera_dtype or self.torch_dtype)
            input_latents = self.vae.encode(video, device=self.device, tiled=tiled, tile_size=(30, 52), tile_stride=(15, 26))
            input_latents = input_latents.to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(input_latents, latents, timestep=self.scheduler.timesteps[0])

        timesteps = self.scheduler.timesteps
        for step, timestep in enumerate(tqdm(timesteps, disable=not progress)):
            camera_ctx["step"] = step
            t = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.dit(latents, t, context_posi, camera_ctx)
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(latents, t, context_nega, camera_ctx)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred - noise_pred_nega)
            latents = self.scheduler.step(noise_pred, timesteps[step], latents)

        video = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=(30, 52), tile_stride=(15, 26))
        return vae_output_to_video(video)
