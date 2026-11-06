"""Load the frozen Wan2.1-T2V-1.3B components from a local folder.

The folder layout is the one of the official Hugging Face repository
``Wan-AI/Wan2.1-T2V-1.3B``::

    Wan2.1-T2V-1.3B/
    ├── diffusion_pytorch_model.safetensors   # DiT
    ├── Wan2.1_VAE.pth                         # VAE
    ├── models_t5_umt5-xxl-enc-bf16.pth        # umT5-XXL text encoder
    └── google/umt5-xxl/                       # tokenizer

Models are instantiated without parameter initialisation, filled with
``load_state_dict(..., assign=True)`` and cast to ``torch_dtype``; this mirrors
DiffSynth-Studio's ``ModelManager`` so the loaded weights are bit-identical.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Union

import torch
from safetensors import safe_open

from .dit import WAN21_T2V_1_3B_CONFIG, WanModel
from .text_encoder import WanTextEncoder
from .vae import WanVideoVAE

WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DIT_FILE = "diffusion_pytorch_model.safetensors"
VAE_FILE = "Wan2.1_VAE.pth"
T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR = "google/umt5-xxl"


@contextmanager
def _skip_parameter_init(device=torch.device("meta")):
    """Create parameters on ``device`` (default: meta) instead of initialising them."""
    old_register_parameter = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter


def _load_file(path: Path) -> dict:
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return {k: f.get_tensor(k) for k in f.keys()}
    return torch.load(path, map_location="cpu", weights_only=True)


def _build(model_cls, state_dict, torch_dtype, device, **kwargs):
    with _skip_parameter_init():
        model = model_cls(**kwargs)
    model = model.eval()
    model.load_state_dict(state_dict, assign=True, strict=True)
    return model.to(dtype=torch_dtype, device=device)


def resolve_wan_dir(wan_dir: Union[str, Path, None] = None, download: bool = True) -> Path:
    """Return a local Wan2.1-T2V-1.3B folder, downloading it from the Hub if needed."""
    if wan_dir is None:
        wan_dir = Path("models") / WAN_REPO_ID
    wan_dir = Path(wan_dir)
    if not (wan_dir / DIT_FILE).exists():
        if not download:
            raise FileNotFoundError(f"Wan2.1-T2V-1.3B not found at {wan_dir}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=WAN_REPO_ID, local_dir=str(wan_dir))
    return wan_dir


def load_dit(wan_dir, torch_dtype=torch.bfloat16, device="cpu") -> WanModel:
    state_dict = _load_file(Path(wan_dir) / DIT_FILE)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("vace")}
    return _build(WanModel, state_dict, torch_dtype, device, **WAN21_T2V_1_3B_CONFIG)


def load_vae(wan_dir, torch_dtype=torch.bfloat16, device="cpu") -> WanVideoVAE:
    state_dict = _load_file(Path(wan_dir) / VAE_FILE)
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    state_dict = {"model." + k: v for k, v in state_dict.items()}
    return _build(WanVideoVAE, state_dict, torch_dtype, device)


def load_text_encoder(wan_dir, torch_dtype=torch.bfloat16, device="cpu") -> WanTextEncoder:
    state_dict = _load_file(Path(wan_dir) / T5_FILE)
    return _build(WanTextEncoder, state_dict, torch_dtype, device)


def tokenizer_path(wan_dir) -> Path:
    return Path(wan_dir) / TOKENIZER_DIR
