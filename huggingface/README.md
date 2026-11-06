---
license: cc-by-nc-sa-4.0
base_model: Wan-AI/Wan2.1-T2V-1.3B
pipeline_tag: text-to-video
library_name: crepe
tags:
  - camera-control
  - video-generation
  - fisheye
  - positional-encoding
  - wan2.1
---

# CRePE: Curved Ray Expectation Positional Encoding (NeurIPS 2026)

Camera-, lens- and orientation-controlled text-to-video generation with a
lightweight **Geometric Attention Adapter** on a frozen
[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

[Paper](https://arxiv.org/abs/2605.12938) · [Code](https://github.com/jinotter3/CRePE) · [Project page](https://crepe-neurips2026.github.io)

CRePE represents every image token by a depth-aware radial distribution along
its Unified Camera Model (UCM) ray and integrates the expected rotary phasor
along the *curved* path this distribution traces when projected into each query
view. The model follows a target camera trajectory under pinhole, wide-angle and
fisheye lenses (horizontal field of view up to 200°) given only a text prompt,
the trajectory and the lens parameters.

## Files

| file | description |
|---|---|
| `crepe.safetensors` | **CRePE**: camera-, lens- and orientation-controlled text-to-video (the camera-controlled videos of the project page) |
| `crepe_mix.safetensors` | **CRePE-Mix**: also accepts an external radial map (camera + radial-map conditioned generation, motion transfer) |

Each file holds only the adapter weights (36.2M parameters, bf16, ~72 MB); the
configuration is stored in the safetensors metadata. The frozen Wan2.1 weights
are loaded from `Wan-AI/Wan2.1-T2V-1.3B`.

**Note.** The CRePE checkpoint used for the paper's experiments was lost. `crepe.safetensors` comes from a
re-implementation of the method, re-trained with the recipe below, so its numbers differ slightly from the
paper's tables.

## Usage

```bash
git clone https://github.com/jinotter3/CRePE && cd CRePE && pip install -r requirements.txt && pip install -e .
hf download jinotter3/CRePE --local-dir checkpoints
python scripts/inference.py --checkpoint checkpoints/crepe.safetensors \
    --input demo/lens_showcase.json --output outputs/lens_showcase --seeds 0
```

```python
from crepe import CRePEPipeline, load_trajectory, save_video

pipe = CRePEPipeline.from_pretrained("checkpoints/crepe.safetensors", device="cuda")
pose = load_trajectory("demo/trajectories/panshot_aQVLCO-OOqM-11-1-no_rot_aug_0.npy")   # (81, 3, 4) camera-to-world
frames = pipe("A majestic red fox sitting on a moss-covered rock ...", pose=pose, x_fov=160.0, xi=1.5, seed=0)
save_video(frames, "fox.mp4", fps=16)
```

Output: 81 frames at 832×480, 16 fps. Inference takes about 4.5 minutes per clip on one B200 and about 18 GB of GPU memory.

## Training (`crepe.safetensors`)

* Data: PanShot train split (47,920 clips synthesised from PanFlow panoramic videos, captions from UCPE), 81 frames at 832×480.
* Frozen Wan2.1-T2V-1.3B; one Geometric Attention Adapter per DiT block (width 192), CRePE in blocks 10–19
  (0-indexed; K = 5 radial breakpoints), UCPE relative-ray encoding elsewhere, absolute orientation map in every block.
* Loss: flow matching + 1e-3 × confidence-weighted radial-distance loss (α = 0.2) on block 15, with pseudo labels
  from [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) (used only during training, not an inference input).
* AdamW, lr 1e-4, 10k steps, bf16, effective batch 16 (2 GPUs × 8). Config: `configs/train_crepe.yaml` in the code repository.

## Limitations

The model inherits the content distribution and the limitations of Wan2.1-T2V-1.3B and of the PanShot training
data. Camera control assumes mostly static scenes; trajectories far outside the training distribution (very fast
motion, unusual scales) are followed less reliably. Generated videos may contain artifacts and should not be
presented as real footage.

## Citation

```bibtex
@inproceedings{jin2026crepe,
  title     = {CRePE: Curved Ray Expectation Positional Encoding for Unified-Camera-Controlled Video Generation},
  author    = {Jin, Seonghyun and Kim, Youngmin and Park, Sunwoo and Ye, Jong Chul},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

The weights in this repository are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (they were trained with pseudo-labels from
UniK3D, CC BY-NC-SA 4.0). The code on GitHub is Apache-2.0; the base model Wan2.1-T2V-1.3B is Apache-2.0 by its authors.
