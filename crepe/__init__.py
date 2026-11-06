"""CRePE: Curved Ray Expectation Positional Encoding for Unified-Camera-Controlled Video Generation."""

__version__ = "1.0.0"

from .adapter import CRePEConfig, GeometricAttentionAdapter, RadialHead, attach_camera_adapters
from .camera_inputs import build_camera_context
from .checkpoint import load_adapter_file, load_adapter_weights, save_adapter
from .crepe_attention import CRePEAttention
from .pipeline import DEFAULT_NEGATIVE_PROMPT, CRePEPipeline
from .trajectory import load_trajectory, resolve_lens_value
from .video_io import save_video

__all__ = [
    "CRePEConfig", "GeometricAttentionAdapter", "RadialHead", "attach_camera_adapters",
    "build_camera_context", "load_adapter_file", "load_adapter_weights", "save_adapter",
    "CRePEAttention", "CRePEPipeline", "DEFAULT_NEGATIVE_PROMPT", "load_trajectory",
    "resolve_lens_value", "save_video",
]
