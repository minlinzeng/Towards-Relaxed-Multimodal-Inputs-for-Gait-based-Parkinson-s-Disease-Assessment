# Project Layout

Core reusable code stays in packages:

- `data/` for readers, loaders, augmentation, preprocessing.
- `models/` for model definitions.
- `optimizers/` for loss functions and multi-objective weighting.
- `learning/` for shared training utilities.
- `const/` for constants and path definitions.
- `mmpose/` for pose extraction utilities.

Executable entrypoints are grouped by purpose:

- `python -m train.train` for the preferred single entrypoint.
- `baselines/` for baseline launchers.
- `experiments/` for TRIP and single-modality launchers.

Legacy root scripts and duplicate experiment folders should be removed once the canonical entrypoints are validated.
