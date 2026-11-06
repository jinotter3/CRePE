<h1 align="center">CRePE: Curved Ray Expectation Positional Encoding<br>for Unified-Camera-Controlled Video Generation</h1>

<p align="center">
  Seonghyun Jin<sup>1*</sup> · Youngmin Kim<sup>1,2*‡</sup> · Sunwoo Park<sup>1*</sup> · Jong Chul Ye<sup>1†</sup><br>
  <sup>1</sup>Kim Jaechul Graduate School of AI, KAIST &nbsp; <sup>2</sup>Korea University<br>
  <sub>* equal contribution &nbsp; † corresponding author &nbsp; ‡ work done during an internship at KAIST</sub><br>
  <b>NeurIPS 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.12938">Paper</a> ·
  <a href="https://crepe-neurips2026.github.io">Project page</a> ·
  <a href="https://huggingface.co/jinotter3/CRePE">Weights</a>
</p>

CRePE is a camera positional encoding for video diffusion transformers. Each
image token is represented by a depth-aware *radial distribution* along its
Unified Camera Model (UCM) ray. For every query view, CRePE integrates the
expected rotary phasor along the **curved path** that this distribution traces
after UCM projection, which makes camera control consistent across pinhole,
wide-angle and fisheye lenses. It is implemented as a lightweight Geometric
Attention Adapter (36.2M parameters) on a frozen Wan2.1-T2V-1.3B.

This repository contains inference and training code and the released
checkpoints for camera-, lens- and orientation-controlled text-to-video
generation.

## Installation

```bash
conda create -n crepe python=3.10 -y && conda activate crepe
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
pip install -r requirements.txt
pip install -e .
# optional, faster attention for the frozen Wan blocks
pip install flash-attn --no-build-isolation
```

The code was tested with Python 3.10, PyTorch 2.7.1 and CUDA 12.8 on NVIDIA B200 GPUs.

## Models

| checkpoint | description |
|---|---|
| `crepe.safetensors` | **CRePE**: camera-, lens- and orientation-controlled text-to-video (the camera-controlled videos of the project page) |
| `crepe_mix.safetensors` | **CRePE-Mix**: additionally accepts an external radial map (camera + radial-map conditioned generation, motion transfer) |

Both are adapter-only files (~72 MB, bf16). Download them, together with the
frozen base model [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
(`hf` is the command-line tool of `huggingface_hub`):

```bash
hf download jinotter3/CRePE --local-dir checkpoints
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan-AI/Wan2.1-T2V-1.3B
```

> **Note.** The CRePE checkpoint used for the paper's experiments was lost. `crepe.safetensors` comes from a
> re-implementation of the method, re-trained with the recipe [below](#recipe-of-the-released-checkpoint), so its
> numbers differ slightly from the paper's tables.

(`CRePEPipeline.from_pretrained` also downloads Wan2.1 automatically if `models/Wan-AI/Wan2.1-T2V-1.3B` is missing.)

## Inference

Generate the 10 lens showcase clips of the project page:

```bash
python scripts/inference.py --checkpoint checkpoints/crepe.safetensors \
    --input demo/lens_showcase.json --output outputs/lens_showcase --seeds 0 1 2
```

One 81-frame 832×480 clip takes about 4.5 minutes on one B200 (50 steps with classifier-free guidance 5.0) and needs about 18 GB of GPU memory (20 GB reserved).
With the settings above the script reproduces the project-page videos bit-exactly.

A single sample:

```bash
python scripts/inference.py --checkpoint checkpoints/crepe.safetensors \
    --trajectory demo/trajectories/panshot_aQVLCO-OOqM-11-1-no_rot_aug_0.npy \
    --x_fov 160 --xi 1.5 --prompt "A majestic red fox sitting on a moss-covered rock ..." --output outputs/fox
```

Python API:

```python
from crepe import CRePEPipeline, load_trajectory, save_video

pipe = CRePEPipeline.from_pretrained("checkpoints/crepe.safetensors", device="cuda")
pose = load_trajectory("demo/trajectories/re10k_ecc2b1b9cb1fa9ad.npy")      # (81, 3, 4) camera-to-world
frames = pipe("A fluffy corgi trotting along a stone path ...", pose=pose, x_fov=170.0, xi=1.8, seed=0)
save_video(frames, "corgi.mp4", fps=16)
```

### Camera + radial-map conditioned generation (CRePE-Mix)

`crepe_mix.safetensors` can follow an externally supplied radial map
(per-pixel camera-centre distance) through the same positional pathway, e.g. to re-render a source video with a new
prompt while keeping its scene geometry and camera motion. This needs [ViPE](https://github.com/nv-tlabs/vipe) (camera
estimation) and [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) (radial map):

```bash
# 81-frame 832x480 clip -> ViPE camera -> UniK3D radial map  (writes samples/my_clip/sample.json)
python scripts/prepare_video.py --input source.mp4 --name my_clip --prompt "A new scene description ..." \
    --vipe_dir /path/to/vipe --resize crop
# generate from noise, conditioned on prompt + camera + radial map (the setting of the paper's DAVIS evaluation)
python scripts/inference.py --checkpoint checkpoints/crepe_mix.safetensors \
    --input samples/my_clip/sample.json --output outputs/my_clip
# optional: start from the source video noised to 75% instead of pure noise (keeps more of its appearance)
python scripts/inference.py --checkpoint checkpoints/crepe_mix.safetensors \
    --input samples/my_clip/sample.json --output outputs/my_clip_sdedit --denoising_strength 0.75
```

For clips with known cameras (e.g. PanShot) skip ViPE with `--pose_npy <pose.npy> --x_fov <deg> --xi <xi> --near_depth <scale>`.
Defaults follow the paper: on rays with a valid UniK3D estimate the predicted radial distribution is replaced by the
map's log-distance with a fixed width σ = 0.1; rays without a valid estimate get a neutral wide interval
(μ = 0, σ = 3) and rays beyond the supported range keep the model's prediction (see `crepe/radial_map.py`).
`--radial_sigma_mode predicted --radial_mu_calibration match_predicted_mean_std` keeps the predicted σ and rescales
the map to the prediction's mean/std; this largely removes the map's effect on the generated geometry.
Note that cuDNN autotuning can make results differ slightly between runs on different GPUs; set
`torch.backends.cudnn.benchmark = False` for repeatable outputs. Training code for CRePE-Mix is not included.

### Camera inputs

* **Trajectory**: 81 camera-to-world matrices `(81, 3, 4)`, OpenCV axes (x right, y down, z forward), world `-y` up.
  Translations use the scene-normalised scale of training. See [`demo/trajectories`](demo/trajectories) and
  [`crepe/trajectory.py`](crepe/trajectory.py) for loaders of RealEstate10K and PanShot poses.
* **Lens**: Unified Camera Model with horizontal field of view `x_fov` (degrees) and mirror parameter `xi`
  (`xi = 0` is a pinhole camera; the showcase uses `x_fov` up to 200° and `xi` up to 2.3).
  Both can be constants or per-frame arrays (`x_fov_schedule` / `xi_schedule` in a sample JSON, see
  [`crepe/trajectory.py`](crepe/trajectory.py)).
* **Orientation**: the absolute pitch and roll of the cameras (relative to world up) are encoded as well, so the
  first camera does not have to be level.

## Training

`crepe.safetensors` is trained on the PanShot dataset of [UCPE](https://github.com/chengzhag/UCPE) with
pseudo radial-distance supervision from [UniK3D](https://github.com/lpiccinelli-eth/UniK3D).

1. **Data.** Download and extract PanShot following the
   [UCPE instructions](https://github.com/chengzhag/UCPE#-panshot-dataset) into `data/UCPE/PanShot`.
   PanFlow's near-plane depths, used to normalise trajectories, are included in
   [`data/UCPE/PanFlow`](data/UCPE/PanFlow) (from UCPE).
2. **Latent and text caches** (Wan2.1 VAE + umT5):
   ```bash
   python scripts/cache_latents.py --data_root data/UCPE --split train      # add --num_shards/--shard_id to parallelise
   ```
3. **Pseudo radial distances** from UniK3D (install it from its repository first):
   ```bash
   python scripts/cache_radial_distance.py --data_root data/UCPE --split train
   ```
4. **Train** (2 GPUs; DeepSpeed ZeRO-1, bf16):
   ```bash
   python scripts/train.py fit --config configs/train_crepe.yaml
   ```
   Adapter-only `.safetensors` files are exported to `logs/<run>/adapters/` every 1000 steps and at the end,
   and can be used directly with `scripts/inference.py`.

### Recipe of the released checkpoint

| | |
|---|---|
| base model | Wan2.1-T2V-1.3B, frozen |
| adapter | one Geometric Attention Adapter per block (width 192, one head), parallel to self-attention, zero-initialised output |
| positional encoding | CRePE in blocks 10–19 (0-indexed; layers 11–20 in the paper) with K = 5 breakpoints, 3 rays per token, frequency base 3 and the radial interval clamped to [−3, 3] in log-distance; UCPE relative-ray encoding in the other blocks; absolute orientation map in all blocks |
| radial head | shared by the CRePE blocks; LayerNorm → Linear(1536, 384) → SiLU → Linear(384, 2) |
| supervision | flow matching + 1e-3 × confidence-weighted radial-distance loss (α = 0.2) on block 15; UniK3D distances divided by the clip's PanFlow near-plane depth; the loss also trains the adapter; skipped for t/1000 < 0.03 |
| data | PanShot train (47,920 clips), 81 frames at 832×480 |
| optimisation | AdamW, lr 1e-4, 10k steps, effective batch 16 (2 GPUs × 8), bf16 |

## Repository layout

```
crepe/
  crepe_attention.py   curved ray expectation RoPE (the core of CRePE)
  relative_ray.py      UCPE relative-ray encoding used outside the CRePE window
  adapter.py           Geometric Attention Adapter, radial head, attachment to the DiT
  geometry.py          Unified Camera Model, ray matrices, absolute orientation map
  camera_inputs.py     per-sample camera context on the latent grid
  pipeline.py          text-to-video sampling
  trajectory.py        trajectory loading and lens schedules
  checkpoint.py        adapter checkpoint format
  wan/                 minimal Wan2.1 T2V model code (from DiffSynth-Studio)
  train/               data, radial supervision and Lightning module
scripts/               inference, training, caching and conversion scripts
configs/               training configuration of the released checkpoint
demo/                  showcase prompts and camera trajectories
tests/                 unit tests (python -m pytest tests)
```

## Citation

```bibtex
@inproceedings{jin2026crepe,
  title     = {CRePE: Curved Ray Expectation Positional Encoding for Unified-Camera-Controlled Video Generation},
  author    = {Jin, Seonghyun and Kim, Youngmin and Park, Sunwoo and Ye, Jong Chul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This work builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1),
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio),
[UCPE](https://github.com/chengzhag/UCPE) (code, PanShot dataset and evaluation protocol),
[RayRoPE](https://github.com/Lucas-707/RayRoPE), [PRoPE](https://github.com/liruilong940607/prope)
and [UniK3D](https://github.com/lpiccinelli-eth/UniK3D). See [NOTICE.md](NOTICE.md) for licenses of included code.

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)); files derived from other projects keep
their original notices (see [NOTICE.md](NOTICE.md)). The model weights (`crepe.safetensors`, `crepe_mix.safetensors`)
are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), as they were trained with
pseudo-labels from UniK3D (CC BY-NC-SA 4.0). The base model Wan2.1 is licensed under Apache-2.0 by its authors.
