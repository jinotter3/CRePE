"""Adapter checkpoint I/O.

Released checkpoints are ``.safetensors`` files that contain only the trainable
adapter weights (about 36M parameters, bf16) with keys relative to the Wan DiT::

    blocks.{i}.camera_adapter.{orientation_proj,q_proj,k_proj,v_proj,out_proj}.{weight,bias}
    radial_head.{norm,fc1,fc2}.{weight,bias}

The adapter configuration is stored as JSON in the safetensors metadata under
``"crepe_config"``. Checkpoints written by the original research code
(DeepSpeed / Lightning keys such as ``pipe.dit.blocks.0.cam_self_attn.q_proj.weight``)
are converted on the fly by :func:`convert_legacy_state_dict`.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .adapter import CRePEConfig, adapter_state_dict

FORMAT_NAME = "crepe-adapter"
FORMAT_VERSION = "1"

_LEGACY_RULES = [
    # Release layout, optionally prefixed by the Lightning module attribute (``dit.``).
    (re.compile(r"^(?:dit\.)?(blocks\.\d+\.camera_adapter\.\w+\.(?:weight|bias)|radial_head\.\w+\.(?:weight|bias))$"), r"\1"),
    (re.compile(r"^(?:pipe\.)?(?:dit\.)?blocks\.(\d+)\.cam_self_attn\.cam_encoder\.(weight|bias)$"),
     r"blocks.\1.camera_adapter.orientation_proj.\2"),
    (re.compile(r"^(?:pipe\.)?(?:dit\.)?blocks\.(\d+)\.cam_self_attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$"),
     r"blocks.\1.camera_adapter.\2.\3"),
    (re.compile(r"^(?:pipe\.)?(?:dit\.)?shared_rayrope_depth_mlp\.0\.(weight|bias)$"), r"radial_head.norm.\1"),
    (re.compile(r"^(?:pipe\.)?(?:dit\.)?shared_rayrope_depth_mlp\.1\.(weight|bias)$"), r"radial_head.fc1.\1"),
    (re.compile(r"^(?:pipe\.)?(?:dit\.)?shared_rayrope_depth_mlp\.3\.(weight|bias)$"), r"radial_head.fc2.\1"),
]


def convert_legacy_state_dict(state_dict: dict) -> dict:
    """Rename research-code adapter keys to the release layout (non-adapter keys are rejected)."""
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    elif "module" in state_dict and isinstance(state_dict["module"], dict):
        state_dict = state_dict["module"]
    out = {}
    for key, value in state_dict.items():
        for pattern, repl in _LEGACY_RULES:
            if pattern.match(key):
                out[pattern.sub(repl, key)] = value
                break
        else:
            raise KeyError(f"Unexpected key in legacy adapter checkpoint: {key}")
    return out


def save_adapter(path: Union[str, Path], state_dict: dict, config: CRePEConfig, extra_metadata: Optional[dict] = None):
    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "base_model": "Wan-AI/Wan2.1-T2V-1.3B",
        "crepe_config": json.dumps(config.to_dict()),
    }
    for k, v in (extra_metadata or {}).items():
        metadata[k] = v if isinstance(v, str) else json.dumps(v)
    tensors = {k: v.detach().contiguous().cpu() for k, v in sorted(state_dict.items())}
    save_file(tensors, str(path), metadata=metadata)


def load_adapter_file(path: Union[str, Path]) -> Tuple[dict, CRePEConfig, dict]:
    """Return ``(state_dict, config, metadata)`` for a release or legacy checkpoint."""
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        config = CRePEConfig.from_dict(json.loads(metadata["crepe_config"])) if "crepe_config" in metadata else CRePEConfig()
        return state_dict, config, metadata
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return convert_legacy_state_dict(raw), CRePEConfig(), {}


def load_adapter_weights(dit, state_dict: dict):
    """Strictly load adapter weights into a DiT that has adapters attached."""
    expected = adapter_state_dict(dit)
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise KeyError(f"Adapter checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    for key, value in state_dict.items():
        if tuple(expected[key].shape) != tuple(value.shape):
            raise ValueError(f"Shape mismatch for {key}: {tuple(value.shape)} vs {tuple(expected[key].shape)}")
    result = dit.load_state_dict(state_dict, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    return dit
