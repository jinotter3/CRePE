# Third-party notices

The CRePE code is released under the Apache License 2.0 (see `LICENSE`); the model weights distributed on
Hugging Face are released under CC BY-NC-SA 4.0. The code includes or adapts code from the projects below, which
keep their original licenses.

| Path | Origin | License |
|---|---|---|
| `crepe/wan/` | [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) port of [Wan2.1](https://github.com/Wan-Video/Wan2.1) (DiT, VAE, umT5 encoder, tokenizer wrapper, flow-matching scheduler) | Apache-2.0 |
| `crepe/geometry.py`, `crepe/camera_inputs.py` | UCM camera geometry, ray matrices and absolute orientation map from [UCPE](https://github.com/chengzhag/UCPE) (pixel grid / UCM unprojection originally from [equilib](https://github.com/haruishi43/equilib), Apache-2.0) | MIT, Copyright (c) 2026 Cheng Zhang |
| `crepe/relative_ray.py` | relative positional encoding from [PRoPE, "Cameras as Relative Positional Encoding"](https://github.com/liruilong940607/prope), as vendored by UCPE (file header: MIT License, Copyright (c) Authors of "Cameras as Relative Positional Encoding") | MIT |
| `crepe/radial_map.py`, `scripts/prepare_video.py` | port of the research code's inference-time Radial MixForcing and external-video preparation | Apache-2.0 (this repository) |
| `crepe/crepe_attention.py` | grew out of a port of [RayRoPE](https://github.com/Lucas-707/RayRoPE) (depth-aware projected RoPE with analytic interval integration) | MIT |
| `data/UCPE/PanFlow/near_plane_depth-*.jsonl` | PanFlow near-plane depths distributed with UCPE | see UCPE / [PanFlow](https://github.com/chengzhag/PanFlow) |
| `demo/trajectories/panshot_*.npy` | camera trajectories of the [PanShot](https://huggingface.co/datasets/chengzhag/PanShot) test split, normalised | CC BY 4.0 (PanShot) |
| `demo/trajectories/re10k_*.npy` | camera trajectories of the [RealEstate10K](https://google.github.io/realestate10k/) test split, normalised | RealEstate10K terms |

External models and tools used but not included:

* **Wan2.1-T2V-1.3B** (frozen base model), Apache-2.0, https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
* **UniK3D** (pseudo radial-distance labels for training and radial maps for CRePE-Mix,
  `scripts/cache_radial_distance.py`, `scripts/prepare_video.py`), CC BY-NC-SA 4.0, https://github.com/lpiccinelli-eth/UniK3D
* **ViPE** (camera estimation for external videos, `scripts/prepare_video.py`), https://github.com/nv-tlabs/vipe
* **DAVIS** videos used for examples: CC BY-NC 4.0, https://davischallenge.org

## MIT License (UCPE, PRoPE, RayRoPE)

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
