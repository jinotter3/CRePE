"""Lightning module that trains the CRePE adapter on a frozen Wan2.1-T2V-1.3B.

The training step reproduces the recipe of the released checkpoints:

* flow-matching loss on cached VAE latents / text contexts with Wan's
  (shift = 5) timestep distribution and its Gaussian loss weighting;
* one timestep per batch; by default one noise tensor shared by all samples
  of the batch (``shared_noise=True``), which is what the original
  implementation did for ``batch_size > 1``;
* radial-distance supervision of the shared radial head at one CRePE layer
  (see :mod:`crepe.train.radial_supervision`), weight ``1e-3``; with the
  released recipe (``configs/train_crepe.yaml``) the targets are scaled by the
  PanFlow near-plane depth, the loss also trains the adapter
  (``radial_head_only: false``) and the confidence weight is ``alpha = 0.2``.

Run with ``precision: bf16-true`` (as in ``configs/``); note that Lightning then
casts the whole batch, including camera poses and lens parameters, to bfloat16.
"""

import os
from pathlib import Path
from typing import Optional

import lightning as pl
import torch
import torch.nn.functional as F

from ..adapter import CRePEConfig, adapter_parameters, adapter_state_dict, attach_camera_adapters
from ..camera_inputs import PATCH_FACTOR, build_camera_context
from ..checkpoint import load_adapter_file, load_adapter_weights, save_adapter
from ..wan import FlowMatchScheduler, load_dit, resolve_wan_dir
from .radial_supervision import RadialSupervisionConfig, radial_supervision_loss


class CRePETrainingModule(pl.LightningModule):
    def __init__(
        self,
        wan_dir: str = "models/Wan-AI/Wan2.1-T2V-1.3B",
        learning_rate: float = 1e-4,
        height: int = 480,
        width: int = 832,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        shared_noise: bool = True,
        crepe_layer_start: int = 10,
        crepe_layer_count: int = 10,
        num_breakpoints: int = 5,
        attn_compress: int = 8,
        point_sample_mode: str = "upper_only",
        radial_supervision: bool = True,
        radial_target_layer: int = 15,
        radial_loss_weight: float = 1e-3,
        radial_min_timestep: float = 0.03,
        radial_head_only: bool = True,
        radial_target_scale: str = "radial_near",
        radial_confidence_alpha: float = 1.0,
        init_adapter: Optional[str] = None,
        export_dir: Optional[str] = None,
        export_every_n_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.crepe_config = CRePEConfig(height=height, width=width, attn_compress=attn_compress,
                                        crepe_layer_start=crepe_layer_start, crepe_layer_count=crepe_layer_count,
                                        num_breakpoints=num_breakpoints, point_sample_mode=point_sample_mode)
        # The original code decoded the prediction with an upper-only log clamp only for the teacher-near targets
        # combined with upper-only interval sampling, and with a symmetric clamp otherwise.
        log_clamp = "upper_only" if radial_target_scale == "radial_near" and point_sample_mode == "upper_only" \
            else "symmetric"
        self.radial_cfg = RadialSupervisionConfig(enabled=radial_supervision, target_layer=radial_target_layer,
                                                  loss_weight=radial_loss_weight, min_timestep=radial_min_timestep,
                                                  head_only=radial_head_only, confidence_alpha=radial_confidence_alpha,
                                                  target_scale=radial_target_scale, log_clamp=log_clamp)
        if self.radial_cfg.enabled and not self.crepe_config.is_crepe_layer(radial_target_layer):
            raise ValueError(f"radial_target_layer={radial_target_layer} is outside the CRePE window")

        dit = load_dit(resolve_wan_dir(wan_dir), torch_dtype=torch.bfloat16, device="cpu")
        attach_camera_adapters(dit, self.crepe_config)
        dit.requires_grad_(False)
        for p in adapter_parameters(dit):
            p.requires_grad_(True)
        if init_adapter is not None:
            state_dict, _, _ = load_adapter_file(init_adapter)
            load_adapter_weights(dit, state_dict)
        self.dit = dit
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        n = sum(p.numel() for p in adapter_parameters(dit))
        print(f"[CRePE] trainable adapter parameters: {n:,}")

    # ------------------------------------------------------------------
    def on_fit_start(self):
        # Every rank must draw its own noise / timesteps (the original runs seeded
        # each process with its local rank). The distributed sampler keeps the
        # shared seed, so the data is still partitioned across ranks.
        seed = int(os.environ.get("PL_GLOBAL_SEED", 0)) + self.global_rank
        torch.manual_seed(seed)

    def training_step(self, batch, batch_idx):
        latents = batch["input_latents"]
        noise_shape = (1 if self.hparams.shared_noise else latents.shape[0],) + tuple(latents.shape[1:])
        noise = torch.randn(noise_shape, device=self.device, dtype=torch.float32).to(latents.dtype)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        loss, logs = self.compute_loss(batch, noise, timestep_id)
        self.log_dict(logs, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def compute_loss(self, batch, noise: torch.Tensor, timestep_id: torch.Tensor):
        """Diffusion (+ radial supervision) loss for given noise and timestep index."""
        latents = batch["input_latents"]
        context = batch["context"]
        camera_ctx = build_camera_context(batch["pose"], batch["x_fov"], batch["xi"],
                                          height=self.hparams.height, width=self.hparams.width,
                                          device=self.device, camera_dtype=torch.bfloat16)
        fraction = float(self.scheduler.timesteps[timestep_id].float().item()) / self.scheduler.num_train_timesteps
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=latents.dtype, device=self.device)

        apply_radial = self.radial_cfg.enabled and fraction >= self.radial_cfg.min_timestep
        if apply_radial:
            camera_ctx["radial_capture"] = {"layer_idx": self.radial_cfg.target_layer,
                                            "head_only": self.radial_cfg.head_only}

        noisy = self.scheduler.add_noise(latents, noise, timestep)
        target = self.scheduler.training_target(latents, noise, timestep)
        pred = self.dit(noisy, timestep, context, camera_ctx,
                        use_gradient_checkpointing=self.hparams.use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=self.hparams.use_gradient_checkpointing_offload)
        diffusion_loss = F.mse_loss(pred.float(), target.float()) * self.scheduler.training_weight(timestep)

        loss = diffusion_loss
        logs = {"train/loss_diffusion": diffusion_loss.detach(), "train/timestep_fraction": fraction,
                "train/radial_applied": float(apply_radial)}
        if apply_radial:
            features = camera_ctx["radial_capture"]["features"]
            scale = batch["radial_near"] if self.radial_cfg.target_scale == "radial_near" else batch["near_depth"]
            radial_loss, metrics = radial_supervision_loss(
                self.dit.radial_head, features, batch["radial_distance"], batch.get("radial_valid"),
                scale, num_cameras=camera_ctx["w2cs"].shape[1],
                patches_y=self.hparams.height // PATCH_FACTOR, patches_x=self.hparams.width // PATCH_FACTOR,
                cfg=self.radial_cfg)
            loss = loss + self.radial_cfg.loss_weight * radial_loss
            logs.update({f"train/{k}": v for k, v in metrics.items()})
        return loss, logs

    def configure_optimizers(self):
        params = [p for p in self.dit.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.learning_rate)

    # ------------------------------------------------------------------
    def export_adapter(self, path):
        save_adapter(path, adapter_state_dict(self.dit), self.crepe_config,
                     extra_metadata={"global_step": int(self.global_step), "hparams": dict(self.hparams)})

    def on_train_batch_end(self, outputs, batch, batch_idx):
        every = self.hparams.export_every_n_steps
        if self.hparams.export_dir and every and self.global_step > 0 and self.global_step % every == 0:
            if self.trainer.is_global_zero:
                out = Path(self.hparams.export_dir)
                out.mkdir(parents=True, exist_ok=True)
                self.export_adapter(out / f"crepe_step{self.global_step:06d}.safetensors")

    def on_train_end(self):
        if self.hparams.export_dir and self.trainer.is_global_zero:
            out = Path(self.hparams.export_dir)
            out.mkdir(parents=True, exist_ok=True)
            self.export_adapter(out / "crepe_last.safetensors")

    def on_save_checkpoint(self, checkpoint):
        # Keep Lightning checkpoints small: the frozen Wan weights are rebuilt at load time.
        for key in list(checkpoint.get("state_dict", {}).keys()):
            if ".camera_adapter." not in key and not key.startswith("dit.radial_head."):
                del checkpoint["state_dict"][key]
