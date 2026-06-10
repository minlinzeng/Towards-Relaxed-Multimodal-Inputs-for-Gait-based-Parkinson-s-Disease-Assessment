import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
try:
    from configs import normalize_dataset_name
except ImportError:
    from train.configs import normalize_dataset_name


class AverageMeter:
    """Track the current value, cumulative sum, count, and running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flatten_skel(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 4:
        batch_size, frame_count, joint_count, coord_count = x.shape
        return x.reshape(batch_size, frame_count, joint_count * coord_count)
    return x


def generate_class_stratified_folds(
    reader: Any,
    dataset: str,
    exclude_subjects: Optional[List[str]] = None,
) -> List[Tuple[List[str], List[str]]]:
    dataset = normalize_dataset_name(dataset)
    exclude_subjects = set(exclude_subjects or [])

    if dataset == "fbg":
        pose_prefixes = {"_".join(key.split("_")[:2]) for key in reader.pose_dict.keys()}
        sensor_prefixes = {"_".join(key.split("_")[:2]) for key in reader.sensor_dict.keys()}
        both_modal = pose_prefixes & sensor_prefixes
        raw = reader.pose_label_dict
        label_dict = {subject: raw[subject] for subject in raw if subject in both_modal and subject not in exclude_subjects}
    elif dataset == "fog":
        label_dict = {subject: reader.labels_dict[subject][0] for subject in reader.labels_dict if subject not in exclude_subjects}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    by_class: Dict[int, List[str]] = {}
    for subject, label in label_dict.items():
        by_class.setdefault(int(label), []).append(subject)

    smallest_class_size = min(len(subjects) for subjects in by_class.values())
    if smallest_class_size == 0:
        raise ValueError("At least one class has zero usable subjects.")

    for key in by_class:
        random.shuffle(by_class[key])

    folds: List[Tuple[List[str], List[str]]] = []
    for index in range(smallest_class_size):
        eval_subjects = [by_class[label][index] for label in sorted(by_class)]
        train_subjects = [subject for subject in label_dict if subject not in eval_subjects]
        folds.append((train_subjects, eval_subjects))
    return folds


def get_branch_class_counts(loader, num_classes: int) -> Tuple[List[int], List[int]]:
    skeleton_counter = Counter()
    sensor_counter = Counter()
    for batch in loader:
        if "label_skeleton" in batch and batch["label_skeleton"] is not None:
            skeleton_counter.update(torch.as_tensor(batch["label_skeleton"]).cpu().tolist())
        if "label_sensor" in batch and batch["label_sensor"] is not None:
            sensor_counter.update(torch.as_tensor(batch["label_sensor"]).cpu().tolist())
    return [skeleton_counter[i] for i in range(num_classes)], [sensor_counter[i] for i in range(num_classes)]


def class_weight_tensor(counts: List[int], device: torch.device) -> torch.Tensor:
    weights = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    return weights / weights.sum() * len(counts)


def print_class_balance(loader, num_classes: int, tag: str = "EVAL", label_names=None):
    skeleton_counter = Counter()
    sensor_counter = Counter()
    for batch in loader:
        if "label_skeleton" in batch and batch["label_skeleton"] is not None:
            skeleton_counter.update(torch.as_tensor(batch["label_skeleton"]).cpu().tolist())
        if "label_sensor" in batch and batch["label_sensor"] is not None:
            sensor_counter.update(torch.as_tensor(batch["label_sensor"]).cpu().tolist())

    names = label_names or [str(index) for index in range(num_classes)]
    total_skel = sum(skeleton_counter.values())
    total_sensor = sum(sensor_counter.values())

    print(f"\n[{tag}] class balance")
    print("class   skel_cnt  skel_%    sens_cnt  sens_%")
    for index, name in enumerate(names):
        skeleton_count = skeleton_counter[index]
        sensor_count = sensor_counter[index]
        skeleton_pct = (skeleton_count / total_skel * 100.0) if total_skel else 0.0
        sensor_pct = (sensor_count / total_sensor * 100.0) if total_sensor else 0.0
        print(f"{name:>5}   {skeleton_count:9d}  {skeleton_pct:6.1f}%   {sensor_count:9d}  {sensor_pct:6.1f}%")

    return [skeleton_counter[i] for i in range(num_classes)], [sensor_counter[i] for i in range(num_classes)]


def count_params(module, trainable_only: bool = True) -> int:
    parameters = [parameter.numel() for parameter in module.parameters() if parameter.requires_grad or not trainable_only]
    return sum(parameters)


def ensemble_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, method: str = "prob_mean") -> torch.Tensor:
    if method == "prob_mean":
        probs_a = F.softmax(logits_a, dim=1)
        probs_b = F.softmax(logits_b, dim=1)
        return (probs_a + probs_b) * 0.5
    if method == "logit_sum":
        return logits_a + logits_b
    raise ValueError(f"Unknown ensemble method: {method}")


def accuracy_from_logits(logits: Optional[torch.Tensor], labels: Optional[torch.Tensor]) -> float:
    if logits is None or labels is None or labels.numel() == 0:
        return 0.0
    predictions = logits.argmax(1)
    return (predictions == labels).float().mean().item() * 100.0


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, predictions = output.topk(maxk, 1, True, True)
        predictions = predictions.t()
        correct = predictions.eq(target.view(1, -1).expand_as(predictions))
        return [
            correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()
            for k in topk
        ]


def compute_class_weights(data_loader) -> List[float]:
    class_counts = Counter()
    total_samples = 0

    for _, targets, _, _ in data_loader:
        class_counts.update(targets.tolist())
        total_samples += len(targets)

    num_classes = len(class_counts)
    class_weights = []
    for index in range(num_classes):
        count = class_counts[index]
        class_weights.append(0.0 if count == 0 else total_samples / (num_classes * count))

    total_weight = sum(class_weights)
    if total_weight == 0:
        return class_weights
    return [weight / total_weight for weight in class_weights]


def _safe_report(trues, preds, label_names=None, name=""):
    trues = list(trues or [])
    preds = list(preds or [])
    if not trues or not preds:
        print(f"\n{name}: (no samples)")
        return
    target_names = label_names if (label_names and len(set(trues)) == len(label_names)) else None
    print(f"\n{name} Report:")
    print(classification_report(trues, preds, digits=2, zero_division=0, target_names=target_names))
    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(trues, preds))


def print_eval_matrix(best: dict, synced: bool, label_names=None, prefix: str = ""):
    if prefix:
        print(prefix)
    if synced:
        _safe_report(best.get("T_sk"), best.get("P_sk"), label_names, "Shared Head")
    else:
        _safe_report(best.get("T_sk"), best.get("P_sk"), label_names, "Skeleton Head")
        _safe_report(best.get("T_se"), best.get("P_se"), label_names, "Sensor   Head")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_training_checkpoint(
    checkpoint_root_path: str,
    epoch: int,
    lr: float,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    best_accuracy: float,
    fold: int,
    latest: bool,
) -> None:
    checkpoint_path_fold = os.path.join(checkpoint_root_path, f"fold{fold}")
    os.makedirs(checkpoint_path_fold, exist_ok=True)
    checkpoint_name = "latest_epoch.pth.tr" if latest else "best_epoch.pth.tr"
    checkpoint_path = os.path.join(checkpoint_path_fold, checkpoint_name)
    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "best_accuracy": best_accuracy,
        },
        checkpoint_path,
    )
