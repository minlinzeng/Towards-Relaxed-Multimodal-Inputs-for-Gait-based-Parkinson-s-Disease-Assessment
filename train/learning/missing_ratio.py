"""Inference-time missing-ratio utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

TWO_MODALITY_SUBSETS: tuple[tuple[str, str, bool, bool], ...] = (
    ("skeleton", "sensor", True, False),
    ("sensor", "skeleton", False, True),
    ("skeleton+sensor", "none", True, True),
)


def parse_ratio_list(text: str) -> list[float]:
    ratios: list[float] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Missing ratio must be in [0, 1], got {value}")
        ratios.append(value)
    if not ratios:
        raise ValueError("At least one missing ratio is required.")
    return ratios


def sample_missing_plan(total: int, ratio: float, n_modalities: int, seed: int) -> np.ndarray:
    """Return a bool array [total, n_modalities] indicating available modalities."""
    enabled = np.ones((total, n_modalities), dtype=bool)
    missing_count = int(round(float(ratio) * total))
    if missing_count <= 0:
        return enabled

    rng = np.random.default_rng(seed)
    selected = rng.choice(total, size=min(missing_count, total), replace=False)
    for sample_idx in selected:
        drop_count = int(rng.integers(1, n_modalities))
        drop = rng.choice(n_modalities, size=drop_count, replace=False)
        enabled[sample_idx, drop] = False
    return enabled


@torch.no_grad()
def eval_two_modality_subset_sync(
    model,
    loader,
    device: torch.device,
    flatten_skel: Callable[[torch.Tensor], torch.Tensor],
    predict_logits: Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor],
    use_skel: bool,
    use_sens: bool,
) -> float:
    """Evaluate a fixed synchronized modality subset for skeleton/sensor data."""
    model.eval()
    corr_sum = n_sum = 0

    for batch in loader:
        skel = flatten_skel(batch["skeleton"]).to(device).float()
        sens = batch["sensor"].to(device).float()
        labels = batch["label_skeleton"].to(device).long()

        if not use_skel:
            skel = torch.zeros_like(skel)
        if not use_sens:
            sens = torch.zeros_like(sens)

        logits = predict_logits(skel, sens, use_skel, use_sens)
        pred = logits.argmax(1)
        corr_sum += (pred == labels).sum().item()
        n_sum += labels.size(0)

    return 100.0 * corr_sum / max(1, n_sum)


@torch.no_grad()
def eval_two_modality_missing_ratio_sync(
    model,
    loader,
    ratio: float,
    seed: int,
    device: torch.device,
    flatten_skel: Callable[[torch.Tensor], torch.Tensor],
    predict_logits: Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor],
) -> float:
    """Evaluate sample-level sync missing-ratio robustness for skeleton/sensor data.

    `predict_logits` receives `(skeleton, sensor, use_skeleton, use_sensor)` and
    returns one logits tensor for the available-modality prediction policy.
    """
    model.eval()
    total = len(loader.dataset)
    plan = sample_missing_plan(total, ratio, n_modalities=2, seed=seed)
    patterns = [(True, True), (True, False), (False, True)]

    corr_sum = n_sum = offset = 0
    for batch in loader:
        batch_n = int(batch["label_skeleton"].size(0))
        batch_plan = plan[offset:offset + batch_n]
        offset += batch_n

        for use_skel, use_sens in patterns:
            sample_mask = np.all(batch_plan == np.array((use_skel, use_sens), dtype=bool), axis=1)
            if not np.any(sample_mask):
                continue

            idx = torch.as_tensor(np.flatnonzero(sample_mask), dtype=torch.long)
            skel = flatten_skel(batch["skeleton"].index_select(0, idx)).to(device).float()
            sens = batch["sensor"].index_select(0, idx).to(device).float()
            labels = batch["label_skeleton"].index_select(0, idx).to(device).long()

            if not use_skel:
                skel = torch.zeros_like(skel)
            if not use_sens:
                sens = torch.zeros_like(sens)

            logits = predict_logits(skel, sens, use_skel, use_sens)
            pred = logits.argmax(1)
            corr_sum += (pred == labels).sum().item()
            n_sum += labels.size(0)

    return 100.0 * corr_sum / max(1, n_sum)
