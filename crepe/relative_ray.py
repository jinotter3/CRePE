# Relative ray encoding (UCPE) for the Geometric Attention Adapter blocks
# outside the CRePE window.
#
# Each token carries the world->ray SE(3) matrix of its UCM viewing ray. Half of
# every attention head is transformed GTA/PRoPE-style by that matrix (queries by
# P^T, keys/values by P^-1, outputs by P) and the other half receives a 2D RoPE
# on the latent patch grid.
#
# Derived from the reference implementation of "Cameras as Relative Positional
# Encoding" (https://github.com/liruilong940607/prope), MIT License,
# Copyright (c) the PRoPE authors. See NOTICE.md.

from functools import partial
from typing import Callable, List, Tuple

import torch

from .geometry import invert_se3


def rope_precompute_coeffs(positions: torch.Tensor, freq_base: float, freq_scale: float, feat_dim: int,
                           dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard 1D RoPE ``(cos, sin)`` of shape ``(1, 1, seqlen, feat_dim // 2)``."""
    assert positions.ndim == 1 and feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device)[None, None, None, :] / num_freqs))
    angles = positions[None, None, :, None] * freqs
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def rope_apply_coeffs(feats: torch.Tensor, coeffs: Tuple[torch.Tensor, torch.Tensor], inverse: bool = False) -> torch.Tensor:
    """Apply RoPE with the 'split' (non-interleaved) layout. ``coeffs`` may cover one image only."""
    cos, sin = coeffs
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2:]
    if not inverse:
        out = [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
    else:
        out = [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
    return torch.cat(out, dim=-1)


def apply_per_token_matrix(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Multiply every 4-vector chunk of each token's features by that token's 4x4 matrix.

    Args:
        feats: ``(B, heads, seqlen, D)`` with ``D % 4 == 0``.
        matrix: ``(B, seqlen, 4, 4)``.
    """
    batch, num_heads, seqlen, feat_dim = feats.shape
    D = matrix.shape[-1]
    assert matrix.shape[1] == seqlen and feat_dim % D == 0
    feats_ = feats.view(batch, num_heads, seqlen, feat_dim // D, D)
    out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
    return out.reshape(feats.shape)


def _apply_block_diagonal(feats: torch.Tensor, func_size_pairs: List[Tuple[Callable, int]]) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    blocks = torch.split(feats, block_sizes, dim=-1)
    return torch.cat([f(b) for f, b in zip(funcs, blocks)], dim=-1)


class RelativeRayEncoding(torch.nn.Module):
    """Relative ray positional encoding with per-token ray matrices.

    Usage::

        enc.prepare(raymats)          # (B, seqlen, 4, 4) world->ray matrices
        q, k, v = enc.apply_q(q), enc.apply_kv(k), enc.apply_kv(v)
        o = attention(q, k, v)
        o = enc.apply_o(o)
    """

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(self, head_dim: int, patches_x: int, patches_y: int, freq_base: float = 100.0, freq_scale: float = 1.0):
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        coeffs_x = rope_precompute_coeffs(torch.tile(torch.arange(patches_x), (patches_y,)),
                                          freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4)
        coeffs_y = rope_precompute_coeffs(torch.repeat_interleave(torch.arange(patches_y), patches_x),
                                          freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4)
        # Patch-grid RoPE tables are recomputed at construction and never stored in checkpoints.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)
        self._fns = None

    def prepare(self, raymats: torch.Tensor):
        """Cache the q/kv/o transforms for ``raymats`` of shape ``(B, seqlen, 4, 4)``."""
        P = raymats
        P_T = P.transpose(-1, -2)
        P_inv = invert_se3(raymats)
        cx = (self.coeffs_x_0, self.coeffs_x_1)
        cy = (self.coeffs_y_0, self.coeffs_y_1)
        h = self.head_dim
        tq = [(partial(apply_per_token_matrix, matrix=P_T), h // 2),
              (partial(rope_apply_coeffs, coeffs=cx), h // 4),
              (partial(rope_apply_coeffs, coeffs=cy), h // 4)]
        tkv = [(partial(apply_per_token_matrix, matrix=P_inv), h // 2),
               (partial(rope_apply_coeffs, coeffs=cx), h // 4),
               (partial(rope_apply_coeffs, coeffs=cy), h // 4)]
        to = [(partial(apply_per_token_matrix, matrix=P), h // 2),
              (partial(rope_apply_coeffs, coeffs=cx, inverse=True), h // 4),
              (partial(rope_apply_coeffs, coeffs=cy, inverse=True), h // 4)]
        self._fns = tuple(partial(_apply_block_diagonal, func_size_pairs=t) for t in (tq, tkv, to))

    def apply_q(self, q):
        return self._fns[0](q)

    def apply_kv(self, kv):
        return self._fns[1](kv)

    def apply_o(self, o):
        return self._fns[2](o)
