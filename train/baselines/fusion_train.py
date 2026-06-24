import random
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, TRAIN_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_processing.preprocess_fbg import PDReader
from data_processing.preprocess_fog import pdfeReader
from feature_encoder import EarlyFusionModel, LateFusionModel, ShareLatentModel, CheapXAttnModel
from data_processing.dataset_cache import load_reader
from data_processing.dataloader_fbg_fog import create_fusion_loaders
from learning.training_common import count_params, generate_class_stratified_folds as shared_generate_class_stratified_folds
from const.path import get_pd_paths
from learning.missing_ratio import (
    TWO_MODALITY_SUBSETS,
    eval_two_modality_missing_ratio_sync,
    eval_two_modality_subset_sync,
    parse_ratio_list,
)
try:
    from configs import FBG_FOG_PARAMS
except ImportError:
    from train.configs import FBG_FOG_PARAMS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = get_pd_paths()
MODALITY_PARAMS = {dataset: dict(params) for dataset, params in FBG_FOG_PARAMS.items()}

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten_skel(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        B, F, J, C = x.shape
        return x.view(B, F, J * C)
    return x

def get_class_counts(loader, num_classes, label_key):
    """
    Count number of samples per class in loader.
    """
    counter = Counter()
    for batch in loader:
        labels = batch[label_key].cpu().tolist()
        counter.update(labels)
    return [counter[i] for i in range(num_classes)]

def create_dataloader(dataset: str, reader, train_subj: list, eval_subj: list):
    """
    Uses the unified efficient loader to return
    (params, train_loader, eval_loader) just like before.
    """
    params = MODALITY_PARAMS[dataset]
    train_loader, eval_loader = create_fusion_loaders(
        dataset       = dataset,
        reader        = reader,
        train_subjects= train_subj,
        eval_subjects = eval_subj,
        batch_size    = params["batch_size"],
        synchronized  = args.synchronized_loading,
        seed          = args.seed,
        num_workers   = 4,
        pad_skel      = params["pose_length"],
        pad_sens      = params["sensor_length"],
    )
    return params, train_loader, eval_loader

def generate_class_stratified_folds(reader, dataset: str):
    return shared_generate_class_stratified_folds(
        reader,
        dataset,
        exclude_subjects=["SUB10", "SUB30", "SUB22"],
    )


def train_one_fold(fold_idx, reader, folds, args):
    params, train_loader, eval_loader = create_dataloader(
        args.dataset, reader, *folds[fold_idx - 1]
    )
    print(" → Eval dataset size:", len(eval_loader.dataset))

    num_classes = params["num_classes"]

    # ——— class‐imbalance weights & criteria ———
    def make_weights(counts):
        t = torch.tensor(counts, dtype=torch.float32, device=device)
        w = 1.0 / (t + 1e-8)
        return w / w.sum() * num_classes

    skel_counts = get_class_counts(train_loader, num_classes, "label_skeleton")
    sens_counts = get_class_counts(train_loader, num_classes, "label_sensor")
    skel_weights = make_weights(skel_counts) if args.wm == "class_wt" else None
    sens_weights = make_weights(sens_counts) if args.wm == "class_wt" else None

    # choose loss(es)
    if args.synchronized_loading and args.fusion_type != "share_latent":
        criterion = nn.CrossEntropyLoss(weight=skel_weights)
    else:
        # for share_latent-sync or any async mode, we need two heads
        criterion_skel = nn.CrossEntropyLoss(weight=skel_weights)
        criterion_sens = nn.CrossEntropyLoss(weight=sens_weights)

    # ——— model instantiation ———
    common_kwargs = dict(
        skeleton_input_dim   = params["skeleton_input_dim"],
        skeleton_output_dim  = params["skeleton_output_dim"],
        sensor_in_channels   = params["sensor_in_channels"],
        sensor_out_channels  = params["sensor_out_channels"],
        sensor_length        = params["sensor_length"],
        shared_out_channels  = params["shared_out_channels"],
        backbone_dim         = params["backbone_dim"],
        num_classes          = params["num_classes"],
        synchronized_loading = args.synchronized_loading,
    )

    if args.fusion_type == "early":
        model = EarlyFusionModel(**common_kwargs).to(device)
    elif args.fusion_type == "late":
        model = LateFusionModel(**common_kwargs).to(device)
    elif args.fusion_type == "share_latent":
        model = ShareLatentModel(
            **common_kwargs,
            taskhead_input_dim = params["backbone_dim"] * params["shared_out_channels"]
        ).to(device)
    elif args.fusion_type == "cheap_xattn":
        model = CheapXAttnModel(**common_kwargs).to(device)
    else:
        raise ValueError(f"Unknown fusion_type: {args.fusion_type}")
    print(f"Total params: {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_avg_acc = 0.0
    best_skel_acc = 0.0
    best_sens_acc = 0.0
    best_state = None

    # storage for best predictions
    if args.synchronized_loading and args.fusion_type != "share_latent":
        best_trues, best_preds = [], []
    else:
        best_trues_skel, best_preds_skel = [], []
        best_trues_sens, best_preds_sens = [], []

    def run_epoch(loader, train=True, collect_preds=False):
        running_loss = 0.0
        total = 0

        # initialize counters
        if args.synchronized_loading and args.fusion_type != "share_latent":
            correct = 0
            trues, preds = [], []
        else:
            correct_sk = correct_se = 0
            trues_skel, preds_skel = [], []
            trues_sens, preds_sens = [], []

        for batch in loader:
            sk = flatten_skel(batch["skeleton"].to(device))
            se = batch["sensor"].to(device)
            y_sk = batch["label_skeleton"].to(device)
            y_se = batch["label_sensor"].to(device)

            # forward + loss
            if args.synchronized_loading and args.fusion_type != "share_latent":
                logits = model(sk, se)
                loss = criterion(logits, y_sk)
            else:
                logits_sk, logits_se = model(sk, se)
                loss_sk = criterion_skel(logits_sk, y_sk)
                loss_se = criterion_sens(logits_se, y_se)
                loss = 0.5 * (loss_sk + loss_se)

            # backward
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total += y_sk.size(0)

            # accuracy + collect
            if args.synchronized_loading and args.fusion_type != "share_latent":
                correct += (logits.argmax(1) == y_sk).sum().item()
                if collect_preds and not train:
                    trues.extend(y_sk.cpu().tolist())
                    preds.extend(logits.argmax(1).cpu().tolist())
            else:
                correct_sk += (logits_sk.argmax(1) == y_sk).sum().item()
                correct_se += (logits_se.argmax(1) == y_se).sum().item()
                if collect_preds and not train:
                    trues_skel.extend(y_sk.cpu().tolist())
                    preds_skel.extend(logits_sk.argmax(1).cpu().tolist())
                    trues_sens.extend(y_se.cpu().tolist())
                    preds_sens.extend(logits_se.argmax(1).cpu().tolist())

        avg_loss = running_loss / len(loader)

        # return metrics
        if args.synchronized_loading and args.fusion_type != "share_latent":
            acc = correct / total * 100
            if collect_preds:
                return avg_loss, acc, trues, preds
            else:
                return avg_loss, acc
        else:
            acc_sk = correct_sk / total * 100
            acc_se = correct_se / total * 100
            if collect_preds:
                return avg_loss, acc_sk, acc_se, trues_skel, preds_skel, trues_sens, preds_sens
            else:
                return avg_loss, acc_sk, acc_se

    # ——— run training ———
    for epoch in range(params["epochs"]):
        model.train()
        # train step
        if args.synchronized_loading and args.fusion_type != "share_latent":
            t_loss, t_acc = run_epoch(train_loader, train=True)
        else:
            t_loss, t_acc_skel, t_acc_sens = run_epoch(train_loader, train=True)

        model.eval()
        with torch.no_grad():
            # eval step
            if args.synchronized_loading and args.fusion_type != "share_latent":
                e_loss, e_acc, trues, preds = run_epoch(
                    eval_loader, train=False, collect_preds=True
                )
                e_avg = e_acc
            else:
                e_loss, e_acc_skel, e_acc_sens, \
                trues_skel, preds_skel, \
                trues_sens, preds_sens = run_epoch(
                    eval_loader, train=False, collect_preds=True
                )
                e_avg = 0.5 * (e_acc_skel + e_acc_sens)

        # update best
        if e_avg > best_avg_acc:
            best_avg_acc = e_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if args.synchronized_loading and args.fusion_type != "share_latent":
                best_trues, best_preds = trues, preds
                best_skel_acc = best_sens_acc = e_acc
            else:
                best_trues_skel, best_preds_skel = trues_skel, preds_skel
                best_trues_sens, best_preds_sens = trues_sens, preds_sens
                best_skel_acc = e_acc_skel
                best_sens_acc = e_acc_sens

        # logging
        if args.synchronized_loading and args.fusion_type != "share_latent":
            print(f"[Fold {fold_idx}][Ep {epoch+1}/{params['epochs']}] "
                  f"Train loss={t_loss:.3f} acc={t_acc:.2f}% | "
                  f"Eval loss={e_loss:.3f} acc={e_acc:.2f}%")
        else:
            print(f"[Fold {fold_idx}][Ep {epoch+1}/{params['epochs']}] "
                  f"Train loss={t_loss:.3f} "
                  f"skel_acc={t_acc_skel:.2f}% sens_acc={t_acc_sens:.2f}% | "
                  f"Eval loss={e_loss:.3f} "
                  f"skel_acc={e_acc_skel:.2f}% sens_acc={e_acc_sens:.2f}% "
                  f"avg_acc={e_avg:.2f}%")

    # ——— final report ———
    if args.synchronized_loading and args.fusion_type != "share_latent":
        print(f"\n*** Fold {fold_idx} Best Acc: {best_avg_acc:.2f}% ***\n")
        print(classification_report(best_trues, best_preds, digits=2, zero_division=0))
    else:
        print(f"\n*** Fold {fold_idx} Best skel={best_skel_acc:.2f}%  sens={best_sens_acc:.2f}%, avg={best_avg_acc:.2f}% ***\n")
        print("Skeleton Head Report:")
        print(classification_report(best_trues_skel, best_preds_skel, digits=2, zero_division=0))
        print("Sensor   Head Report:")
        print(classification_report(best_trues_sens, best_preds_sens, digits=2, zero_division=0))

    if args.eval_missing_ratios and args.synchronized_loading and best_state is not None:
        model.load_state_dict(best_state, strict=True)

        def predict_logits(skel, sens, use_skel, use_sens):
            if args.fusion_type != "share_latent":
                return model(skel, sens)

            log_skel, log_sens = model(skel, sens)
            probs = []
            if use_skel:
                probs.append(F.softmax(log_skel, dim=1))
            if use_sens:
                probs.append(F.softmax(log_sens, dim=1))
            return sum(probs) / len(probs)

        for available, missing, use_skel, use_sens in TWO_MODALITY_SUBSETS:
            score = eval_two_modality_subset_sync(
                model, eval_loader, device, flatten_skel, predict_logits, use_skel, use_sens
            )
            args._subset_scores[available].append(float(score))
            print(
                f"[SYNC][available={available}][missing={missing}] "
                f"acc={score:5.2f}% over {len(eval_loader.dataset)} samples"
            )

        for ratio_idx, ratio in enumerate(args._missing_ratios):
            repeat_scores = []
            for rep in range(args.missing_ratio_repeats):
                ratio_seed = args.seed * 100000 + fold_idx * 1000 + ratio_idx * 100 + rep
                score = eval_two_modality_missing_ratio_sync(
                    model, eval_loader, ratio, ratio_seed, device, flatten_skel, predict_logits
                )
                repeat_scores.append(float(score))
            repeat_arr = np.array(repeat_scores, dtype=float)
            args._missing_ratio_scores[ratio].append(float(repeat_arr.mean()))
            print(
                f"[SYNC][missing_ratio={ratio:.2f}] "
                f"fold_mean={repeat_arr.mean():5.2f}% ± {repeat_arr.std():4.2f}% "
                f"over {args.missing_ratio_repeats} repeats"
            )

    print("-" * 60 + "\n")
    return best_avg_acc, best_skel_acc, best_sens_acc


def main(args):
    # 1) Reproducibility
    set_random_seed(args.seed)
    print("Arguments:", args)
    print("Dataset params:", MODALITY_PARAMS[args.dataset])
    args._missing_ratios = parse_ratio_list(args.missing_ratios)
    args._missing_ratio_scores = {ratio: [] for ratio in args._missing_ratios}
    args._subset_scores = {available: [] for available, *_ in TWO_MODALITY_SUBSETS}

    # 2) Instantiate the appropriate reader
    reader = load_reader(args.dataset, rebuild=args.rebuild_cache)

    # 3) Build stratified folds at the subject level
    folds = generate_class_stratified_folds(reader, args.dataset)

    # 4) Header
    print(f"\n>>> Running {args.fusion_type.upper()}-FUSION baseline on {args.dataset.upper()} <<<\n")

    # 5) Loop over folds and train
    avg_accs = []
    skel_accs = []
    sens_accs = []

    for fold_idx in range(1, len(folds) + 1):
        avg_acc, skel_acc, sens_acc = train_one_fold(fold_idx, reader, folds, args)
        avg_accs.append(avg_acc)
        skel_accs.append(skel_acc)
        sens_accs.append(sens_acc)

    # 6) Compute mean accuracy metrics
    avg_accs = np.array(avg_accs)
    skel_accs = np.array(skel_accs)
    sens_accs = np.array(sens_accs)

    mean_avg = avg_accs.mean()
    mean_skel = skel_accs.mean()
    mean_sens = sens_accs.mean()

    # 7) Print detailed per-fold summary
    print("\n" + "=" * 55)
    print(" ACCURACY PER FOLD ")
    print("=" * 55)
    for i in range(len(avg_accs)):
        print(f"Fold {i+1}: Avg={avg_accs[i]:.2f}%, Skel={skel_accs[i]:.2f}%, Sensor={sens_accs[i]:.2f}%")

    print(f"\n→ Mean Eval Acc: {mean_avg:.2f}%")
    print(f"→ mean skel={mean_skel:.2f}%, sensor={mean_sens:.2f}%, avg={(mean_skel + mean_sens)/2:.2f}%")
    print("=" * 55 + "\n")

    if args.eval_missing_ratios and args.synchronized_loading and all(args._missing_ratio_scores.values()):
        if all(args._subset_scores.values()):
            print("\n=== Modality-subset accuracy at best epoch (sync deterministic mask) ===")
            print("Policy: deterministic subset evaluation; missing modalities are zero-filled.")
            for available, missing, *_ in TWO_MODALITY_SUBSETS:
                subset_arr = np.array(args._subset_scores[available], dtype=float)
                print(
                    f"[available={available}][missing={missing}] "
                    f"{subset_arr.mean():5.2f}% ± {subset_arr.std():4.2f}%  over {len(subset_arr)} folds"
                )

        print("\n=== Missing-ratio accuracy at best epoch (sync inference stress test) ===")
        print(f"Policy: sample-level Method B, {args.missing_ratio_repeats} repeats per fold; selected samples miss one modality.")
        for ratio in args._missing_ratios:
            ratio_arr = np.array(args._missing_ratio_scores[ratio], dtype=float)
            print(f"[ratio={ratio:.2f}] {ratio_arr.mean():5.2f}% ± {ratio_arr.std():4.2f}%  over {len(ratio_arr)} folds")

    # 8) Show which subjects were held out in each fold
    print("=" * 40)
    print(" EVAL SUBJECTS PER FOLD ")
    print("=" * 40)
    for i, (_, eval_subj) in enumerate(folds, start=1):
        print(f" Fold {i:>2}: {eval_subj}")
    print()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="fbg", choices=["fbg","fog"])
        parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild the dataset .pkl cache before training")
        parser.add_argument("--seed", type=int, default=43)
        parser.add_argument("--wm", choices=["ce", "class_wt"], default="ce")
        parser.add_argument("--fusion_type", choices=["early","late","share_latent","cheap_xattn"], default="cheap_xattn", help="early: concat→backbone→two-heads  late: two branches→fuse logits")
        parser.add_argument("--synchronized_loading", action="store_true", help="use time-synchronized sampling for both train and eval")
        parser.add_argument("--eval_missing_ratios", action="store_true",
                            help="Run sync-only sample-level inference-time missing-ratio evaluation after the best epoch.")
        parser.add_argument("--missing_ratios", type=str, default="0.2,0.4,0.6,0.8",
                            help="Comma-separated missing ratios for --eval_missing_ratios.")
        parser.add_argument("--missing_ratio_repeats", type=int, default=5,
                            help="Random repeats per fold for --eval_missing_ratios.")
        args = parser.parse_args()
        main(args)
