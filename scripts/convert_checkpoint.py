#!/usr/bin/env python
"""Convert a research-code checkpoint into a release ``.safetensors`` adapter file.

Accepts either a DeepSpeed ``mp_rank_00_model_states.pt`` (only the trainable
parameters listed in ``param_shapes`` are kept) or a slim ``{"state_dict": ...}``
file with legacy keys.

    python scripts/convert_checkpoint.py <input.pt> <output.safetensors> [--name crepe] [--point_sample_mode clamped]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crepe.adapter import CRePEConfig  # noqa: E402
from crepe.checkpoint import convert_legacy_state_dict, save_adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--name", default=None, help="model name stored in the metadata")
    parser.add_argument("--metadata", default=None, help="JSON string with extra metadata")
    parser.add_argument("--point_sample_mode", default="upper_only", choices=["upper_only", "clamped"],
                        help="radial-interval clamping used by the checkpoint (released checkpoints: clamped)")
    args = parser.parse_args()

    raw = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    if "module" in raw and "param_shapes" in raw:
        groups = raw["param_shapes"] if isinstance(raw["param_shapes"], list) else [raw["param_shapes"]]
        keys = set().union(*[set(g.keys()) for g in groups])
        raw = {k: raw["module"][k] for k in sorted(keys)}
    state_dict = convert_legacy_state_dict(raw)
    extra = json.loads(args.metadata) if args.metadata else {}
    if args.name:
        extra["name"] = args.name
    save_adapter(args.output, state_dict, CRePEConfig(point_sample_mode=args.point_sample_mode), extra_metadata=extra)
    n = sum(v.numel() for v in state_dict.values())
    print(f"{len(state_dict)} tensors, {n / 1e6:.2f}M params -> {args.output}")


if __name__ == "__main__":
    main()
