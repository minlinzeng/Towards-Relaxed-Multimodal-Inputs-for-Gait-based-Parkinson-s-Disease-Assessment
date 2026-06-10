import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from feature_encoder import MultiModalMultiTaskModel, SensorModalityModel, SkelModalityModel
from learning.optimizers.classification_losses import GCLLoss, LDAMLoss
try:
    from configs import normalize_dataset_name
except ImportError:
    from train.configs import normalize_dataset_name


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_skel(x):
    if x.dim() == 4:
        batch_size, frame_count, joint_count, coord_count = x.shape
        return x.view(batch_size, frame_count, joint_count * coord_count)
    return x


def choose_model(args, params, device):
    if args.modality == "skeleton":
        return SkelModalityModel(
            skeleton_input_dim=params["skeleton_input_dim"],
            skeleton_output_dim=params["skeleton_output_dim"],
            sensor_out_channels=params["skeleton_output_dim"],
            shared_out_channels=params["shared_out_channels"],
            backbone_dim=params["backbone_dim"],
            taskhead_input_dim=params["taskhead_input_dim"],
            num_classes=params["num_classes"],
        ).to(device)

    if args.modality == "sensor":
        return SensorModalityModel(
            sensor_in_channels=params["sensor_in_channels"],
            sensor_out_channels=params["sensor_out_channels"],
            sensor_length=params["sensor_length"],
            shared_out_channels=params["shared_out_channels"],
            backbone_dim=params["backbone_dim"],
            taskhead_input_dim=params["taskhead_input_dim"],
            num_classes=params["num_classes"],
        ).to(device)

    return MultiModalMultiTaskModel(
        skeleton_input_dim=params["skeleton_input_dim"],
        skeleton_output_dim=params["skeleton_output_dim"],
        sensor_in_channels=params["sensor_in_channels"],
        sensor_out_channels=params["sensor_out_channels"],
        sensor_length=params["sensor_length"],
        shared_out_channels=params["shared_out_channels"],
        backbone_dim=params["backbone_dim"],
        taskhead_input_dim=params["taskhead_input_dim"],
        num_classes=params["num_classes"],
        use_norm=args.use_norm_and_cos,
        use_cosine=args.use_norm_and_cos,
        synchronized_loading=args.synchronized_loading,
    ).to(device)


def get_branch_class_counts(loader, num_classes):
    skeleton_counter = Counter()
    sensor_counter = Counter()
    for batch in loader:
        if "label_skeleton" in batch:
            skeleton_counter.update(batch["label_skeleton"].cpu().tolist())
        if "label_sensor" in batch:
            sensor_counter.update(batch["label_sensor"].cpu().tolist())

    skeleton_counts = [skeleton_counter[i] for i in range(num_classes)]
    sensor_counts = [sensor_counter[i] for i in range(num_classes)]
    print(f"Skeleton counts: {skeleton_counts}, Sensor counts: {sensor_counts}")
    return skeleton_counts, sensor_counts


def generate_class_stratified_folds(reader, dataset):
    dataset = normalize_dataset_name(dataset)
    if dataset == "fbg":
        pose_prefixes = {"_".join(key.split("_")[:2]) for key in reader.pose_dict.keys()}
        sensor_prefixes = {"_".join(key.split("_")[:2]) for key in reader.sensor_dict.keys()}
        both_modal = pose_prefixes & sensor_prefixes
        raw = reader.pose_label_dict
        label_dict = {subject: raw[subject] for subject in raw if subject in both_modal}
    elif dataset == "fog":
        label_dict = {
            subject: reader.labels_dict[subject][0]
            for subject in reader.labels_dict
            if subject not in ("SUB10", "SUB30", "SUB22")
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    class_to_subjects = defaultdict(list)
    for subject, label in label_dict.items():
        class_to_subjects[label].append(subject)

    fold_count = min(len(subjects) for subjects in class_to_subjects.values())
    if fold_count == 0:
        raise ValueError("Need at least one subject per class with both modalities for FBG")

    balanced = {}
    for label, subjects in class_to_subjects.items():
        if len(subjects) > fold_count:
            subjects = random.sample(subjects, k=fold_count)
        random.shuffle(subjects)
        balanced[label] = subjects

    folds = []
    for index in range(fold_count):
        eval_subjects = [balanced[label][index] for label in sorted(balanced)]
        train_subjects = [subject for subject in label_dict if subject not in eval_subjects]
        folds.append((train_subjects, eval_subjects))
    return folds


def make_inv_freq_weights(counts, device):
    weights = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    return weights / weights.sum() * len(counts)


def compute_log_based_weights(counts, div, device):
    counts_np = np.array(counts, dtype=np.float32)
    max_count = counts_np.max()
    raw = np.log(max_count / counts_np + 0.01) / div
    raw = np.clip(raw, a_min=0.0, a_max=None)
    if raw.sum() > 0:
        raw = raw / raw.sum() * len(raw)
    return torch.tensor(raw, dtype=torch.float32, device=device)


def weighted_branch_loss(logits, labels, method, class_counts, train=True):
    class_count = logits.shape[1]
    if method == "ce":
        return nn.CrossEntropyLoss()(logits, labels), {}
    if method == "class_wt":
        weights = 1.0 / (torch.tensor(class_counts, device=labels.device) + 1e-8)
        weights = weights / weights.sum() * class_count
        return nn.CrossEntropyLoss(weight=weights)(logits, labels), {"weights": weights}
    raise ValueError(method)


def build_branch_losses(args, skeleton_counts, sensor_counts, device):
    ldam_skel = ldam_sens = None
    gcl_skel = gcl_sens = None
    drw_weights = {"skeleton": None, "sensor": None}

    if args.wm.lower() == "ldam":
        skeleton_weights = make_inv_freq_weights(skeleton_counts, device)
        sensor_weights = make_inv_freq_weights(sensor_counts, device)
        ldam_skel = LDAMLoss(
            cls_num_list=skeleton_counts,
            max_m=args.ldam_m,
            weight=skeleton_weights,
            s=args.ldam_s,
        ).to(device)
        ldam_sens = LDAMLoss(
            cls_num_list=sensor_counts,
            max_m=args.ldam_m,
            weight=sensor_weights,
            s=args.ldam_s,
        ).to(device)

    if args.wm.lower() == "gcl":
        drw_weights["skeleton"] = make_inv_freq_weights(skeleton_counts, device)
        drw_weights["sensor"] = make_inv_freq_weights(sensor_counts, device)
        gcl_skel = GCLLoss(
            cls_num_list=skeleton_counts,
            m=args.gcl_m,
            s=args.gcl_s,
            noise_mul=args.noise_mul,
            weight=None,
        ).to(device)
        gcl_sens = GCLLoss(
            cls_num_list=sensor_counts,
            m=args.gcl_m,
            s=args.gcl_s,
            noise_mul=args.noise_mul,
            weight=None,
        ).to(device)

    return ldam_skel, ldam_sens, gcl_skel, gcl_sens, drw_weights


def apply_gcl_drw(args, epoch, fold_idx, gcl_skel, gcl_sens, drw_weights):
    if args.wm.lower() != "gcl" or epoch != args.drw_warmup:
        return
    print(f"[Fold {fold_idx}] DRW: applying class weights at epoch {epoch + 1}")
    gcl_skel.weight = drw_weights["skeleton"]
    gcl_sens.weight = drw_weights["sensor"]


def save_loss_curve(args, fold_idx, train_losses, val_losses):
    if not args.save_loss_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold_idx} Loss Curves")
    plt.legend()
    plt.tight_layout()
    out_dir = os.path.join("loss_plots", f"fold_{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{args.dataset}_{args.modality}_{args.wm}_loss_curve.png"))
    plt.close()
