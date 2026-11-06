# Demo inputs

* `lens_showcase.json`: the 10 prompts / trajectories / lenses of the project-page lens showcase.
  `python scripts/inference.py --checkpoint checkpoints/crepe.safetensors --input demo/lens_showcase.json --seeds 0 1 2`
  reproduces those videos (seeds 0-2).
* `radial_prompts.json`: prompts of the radial-map examples (CRePE-Mix).

## Trajectories

`trajectories/*.npy` are `(81, 3, 4)` camera-to-world matrices (OpenCV axes, world `-y` up), already normalised
exactly as for training and evaluation, stored in the dtype of the source data:

* `panshot_<clip>.npy`: PanShot test split ([chengzhag/PanShot](https://huggingface.co/datasets/chengzhag/PanShot), CC BY 4.0).
  Translations divided by the clip's PanFlow near-plane depth; the world is rotated about the vertical axis so the
  first camera looks along the y-z plane (`crepe.trajectory.normalize_panshot_pose`).
* `re10k_<id>.npy`: RealEstate10K test split ([google.github.io/realestate10k](https://google.github.io/realestate10k/)).
  Expressed relative to the first camera and scaled by `0.95697` (`crepe.trajectory.load_re10k_trajectory`).

To use your own trajectory, save an `(81, 3, 4)` array with a similar scale (camera speeds of the order of the
examples), or pass a RealEstate10K camera `.txt` file directly.
