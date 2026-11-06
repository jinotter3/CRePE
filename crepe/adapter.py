"""Geometric Attention Adapter.

The frozen Wan2.1 DiT is extended with one lightweight camera-aware attention
branch per block, run in parallel with the block's own self-attention::

    residual = SelfAttn(x) + Adapter(x, camera)

Inside the adapter, the token is first offset by an embedding of the absolute
orientation map (per-pixel gravity direction + latitude), then projected to a
narrow attention space (``dim / attn_compress``). Blocks inside the CRePE
window use :class:`~crepe.crepe_attention.CRePEAttention`, whose radial
distributions are predicted by a :class:`RadialHead` shared by all window
blocks; the remaining blocks use relative ray encoding (UCPE). The output
projection is zero-initialised so the adapter starts as an identity.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from .crepe_attention import DEFAULT_RAY_OFFSETS, CRePEAttention
from .radial_map import apply_radial_condition, condition_active
from .relative_ray import RelativeRayEncoding
from .wan.dit import WanModel, flash_attention


@dataclass
class CRePEConfig:
    """Architecture of the released CRePE adapter (Wan2.1-T2V-1.3B, 832 x 480)."""

    height: int = 480
    width: int = 832
    attn_compress: int = 8               # adapter width = dim / attn_compress (192 = one head)
    crepe_layer_start: int = 10          # CRePE window = blocks [start, start + count)
    crepe_layer_count: int = 10
    num_breakpoints: int = 5             # K breakpoints per radial interval
    point_sample_mode: str = "upper_only"  # radial-interval clamping (released checkpoints: "clamped")
    crepe_freq_base: float = 3.0
    relray_freq_base: float = 100.0
    radial_head_hidden_ratio: float = 0.25
    radial_init_mu: float = 0.0          # radial head bias init (log-distance centre)
    radial_init_sigma: float = 3.0       # radial head bias init (log half-width)
    orientation_dim: int = 3             # absolute orientation map: up (2) + latitude (1)
    ray_offsets: Tuple[Tuple[float, float], ...] = DEFAULT_RAY_OFFSETS

    def to_dict(self):
        d = asdict(self)
        d["ray_offsets"] = [list(o) for o in self.ray_offsets]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "ray_offsets" in d:
            d["ray_offsets"] = tuple(tuple(o) for o in d["ray_offsets"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_crepe_layer(self, idx: int) -> bool:
        return self.crepe_layer_start <= idx < self.crepe_layer_start + self.crepe_layer_count


class RadialHead(nn.Module):
    """Predicts a token's radial distribution ``(mu, sigma)`` (log distance) from its feature."""

    def __init__(self, dim: int, hidden: int, init_mu: float = 0.0, init_sigma: float = 3.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, 2)
        with torch.no_grad():
            nn.init.zeros_(self.fc2.weight)
            self.fc2.bias.copy_(torch.tensor([init_mu, init_sigma], dtype=self.fc2.bias.dtype))

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


class GeometricAttentionAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_heads: int,
        patches_x: int,
        patches_y: int,
        config: CRePEConfig,
        layer_idx: int,
        use_crepe: bool,
        radial_head: Optional[RadialHead] = None,
    ):
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.layer_idx = layer_idx
        self.use_crepe = use_crepe

        self.orientation_proj = nn.Linear(config.orientation_dim, dim)
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        if use_crepe:
            if radial_head is None:
                raise ValueError("CRePE layers need the shared radial head")
            # Shared across window blocks; owned (and checkpointed) by the DiT.
            object.__setattr__(self, "radial_head", radial_head)
            self.crepe_attn = CRePEAttention(
                head_dim=self.head_dim, patches_x=patches_x, patches_y=patches_y,
                image_width=config.width, image_height=config.height,
                num_breakpoints=config.num_breakpoints, freq_base=config.crepe_freq_base,
                ray_offsets=config.ray_offsets, point_sample_mode=config.point_sample_mode,
            )
        else:
            self.relray = RelativeRayEncoding(self.head_dim, patches_x, patches_y, freq_base=config.relray_freq_base)

    def forward(self, x: torch.Tensor, camera_ctx: dict) -> torch.Tensor:
        B, T, D = x.shape
        y = self.orientation_proj(camera_ctx["orientation"])
        if y.shape[1] != T:
            y = repeat(y, "b f d -> b (f hw) d", hw=T // y.shape[1])
        x = x + y

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_crepe:
            radial = self.radial_head(x)
            capture = camera_ctx.get("radial_capture")
            if capture is not None and capture.get("layer_idx") == self.layer_idx:
                # Radial-distance supervision (training): keep the head input so the loss can
                # re-run the head, on a detached copy when only the head is supervised.
                capture["features"] = x.detach() if capture.get("head_only", True) else x
            condition = camera_ctx.get("radial_map")
            if condition is not None and condition_active(condition["config"], camera_ctx.get("step")):
                # External radial map (CRePE-Mix): replace the predicted distribution where the map is valid.
                radial = apply_radial_condition(radial, condition, condition["config"])
            out = self.crepe_attn(q, k, v, radial, camera_ctx)
        else:
            self.relray.prepare(camera_ctx["raymats"])
            q = rearrange(self.relray.apply_q(q), "b h t d -> b t (h d)")
            k = rearrange(self.relray.apply_kv(k), "b h t d -> b t (h d)")
            v = rearrange(self.relray.apply_kv(v), "b h t d -> b t (h d)")
            out = flash_attention(q, k, v, num_heads=self.num_heads)
            out = rearrange(out, "b t (h d) -> b h t d", h=self.num_heads)
            out = self.relray.apply_o(out)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)


def attach_camera_adapters(dit: WanModel, config: CRePEConfig, patch_factor: int = 16) -> WanModel:
    """Attach a Geometric Attention Adapter to every DiT block (in place).

    New modules are created in float32 on CPU and should be loaded / cast
    afterwards (``load_adapter_weights`` + ``.to(dtype, device)``).
    """
    patches_x = config.width // patch_factor
    patches_y = config.height // patch_factor
    dim = dit.dim
    num_blocks = len(dit.blocks)
    end = config.crepe_layer_start + config.crepe_layer_count
    if not (0 <= config.crepe_layer_start < end <= num_blocks):
        raise ValueError(f"invalid CRePE window [{config.crepe_layer_start}, {end}) for {num_blocks} blocks")

    radial_head = RadialHead(dim, max(16, int(dim * config.radial_head_hidden_ratio)),
                             init_mu=config.radial_init_mu, init_sigma=config.radial_init_sigma)
    dit.radial_head = radial_head
    for idx, block in enumerate(dit.blocks):
        block.camera_adapter = GeometricAttentionAdapter(
            dim=dim,
            attn_dim=dim // config.attn_compress,
            num_heads=block.num_heads // config.attn_compress,
            patches_x=patches_x,
            patches_y=patches_y,
            config=config,
            layer_idx=idx,
            use_crepe=config.is_crepe_layer(idx),
            radial_head=radial_head if config.is_crepe_layer(idx) else None,
        )
    dit.crepe_config = config
    return dit


def adapter_parameters(dit: WanModel):
    """Trainable parameters: all camera adapters + the shared radial head."""
    params = list(dit.radial_head.parameters())
    for block in dit.blocks:
        params += list(block.camera_adapter.parameters())
    return params


def adapter_state_dict(dit: WanModel) -> dict:
    """State dict restricted to the adapter (keys relative to the DiT)."""
    return {k: v for k, v in dit.state_dict().items() if ".camera_adapter." in k or k.startswith("radial_head.")}
