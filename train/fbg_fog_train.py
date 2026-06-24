"""FBG/FoG multitask training entry point.

This file keeps only the experiment flow:
1. process_batch: one forward/loss/backward step
2. run_epoch: repeat the batch step over a loader
3. train_one_fold: build data/model/losses and train one subject fold
4. main: run all folds and requested modalities

Static settings live in configs.py. Reusable model/loss/fold helpers live in
utilities.py. Data preprocessing and augmentation live under data_processing/.
"""

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_processing.dataset_cache import load_reader
from data_processing.dataloader_fbg_fog import create_fusion_loaders
try:
    from configs import FBG_FOG_PARAMS, FBG_FOG_PARAM_PRESETS, normalize_dataset_name
except ImportError:
    from train.configs import FBG_FOG_PARAMS, FBG_FOG_PARAM_PRESETS, normalize_dataset_name
from learning.optimizers.multitask_weighting import CAGrad
from learning.missing_ratio import (
    TWO_MODALITY_SUBSETS,
    eval_two_modality_missing_ratio_sync,
    eval_two_modality_subset_sync,
    parse_ratio_list,
)
from learning.training_common import count_params
from sklearn.metrics import classification_report, confusion_matrix
from utilities import (
    apply_gcl_drw,
    build_branch_losses,
    choose_model,
    flatten_skel,
    generate_class_stratified_folds,
    get_branch_class_counts,
    save_loss_curve,
    set_random_seed,
    weighted_branch_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_param_preset(args):
    """Apply named FBG/FoG reproduction settings to parsed CLI args."""
    args._param_overrides = {}
    preset_name = (getattr(args, "param_preset", "") or "").strip()
    if not preset_name:
        return

    preset = FBG_FOG_PARAM_PRESETS.get(preset_name)
    if preset is None:
        available = ", ".join(sorted(FBG_FOG_PARAM_PRESETS)) or "(none)"
        raise ValueError(f"Unknown --param_preset '{preset_name}'. Available: {available}")

    expected_dataset = preset.get("dataset")
    if expected_dataset and args.dataset != expected_dataset:
        raise ValueError(
            f"--param_preset {preset_name} is for dataset={expected_dataset}, "
            f"but current dataset={args.dataset}."
        )
    if preset.get("sync_required") and not args.synchronized_loading:
        raise ValueError(f"--param_preset {preset_name} requires --synchronized_loading.")

    for key, value in preset.get("args", {}).items():
        setattr(args, key, value)
    args._param_overrides.update(preset.get("train_params", {}))

    arg_text = ", ".join(f"{k}={v}" for k, v in sorted(preset.get("args", {}).items()))
    train_text = ", ".join(f"{k}={v}" for k, v in sorted(args._param_overrides.items()))
    print(f"[PRESET] Applied {preset_name}: args({arg_text}); train_params({train_text})")


def apply_train_param_overrides(args):
    """Apply explicit CLI overrides for config-level training parameters."""
    overrides = getattr(args, "_param_overrides", {})
    if getattr(args, "epochs", None) is not None:
        overrides["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        overrides["batch_size"] = args.batch_size
    args._param_overrides = overrides


# ---------------------------------------------------------------------------
# Batch Step
# ---------------------------------------------------------------------------
def process_batch(
    batch,
    model,
    optimizer,
    args,
    sk_counts,
    se_counts,
    ldam_skel,
    ldam_sens,
    gcl_skel,
    gcl_sens,
    grad_method,
    device,
    train: bool,
):
    """Run one minibatch and optionally update model parameters.

    Returns:
        loss_value, correct_skeleton, correct_sensor, batch_size
    """
    skeleton, sensor = None, None
    if "skeleton" in batch:
        skeleton = flatten_skel(batch["skeleton"].to(device)).float()
        y_skel   = batch["label_skeleton"].to(device).long()
    if "sensor" in batch:
        sensor   = batch["sensor"].to(device).float()
        y_sens   = batch["label_sensor"].to(device).long()

    # Forward pass. Multimodal models return one head per branch.
    consistency = None
    if args.modality == "multimodal":
        p_skel, p_sens = model(skeleton, sensor)

        # In synchronized multimodal mode, the two heads see paired samples.
        # The symmetric KL term encourages their predictions to agree.
        if args.synchronized_loading:
            logp = F.log_softmax(p_skel, dim=1)
            q    = F.softmax(p_sens, dim=1)
            kl1  = F.kl_div(logp, q, reduction="batchmean")

            logq = F.log_softmax(p_sens, dim=1)
            p    = F.softmax(p_skel, dim=1)
            kl2  = F.kl_div(logq, p, reduction="batchmean")
            consistency = kl1 + kl2
    elif args.modality == "skeleton":
        p_skel, p_sens = model(skeleton), None
    elif args.modality == "sensor":
        p_skel, p_sens = None, model(sensor)
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    # Loss selection. CE/class_wt are computed directly; LDAM/GCL modules are
    # constructed once per fold because they depend on class counts.
    if args.wm == "ldam":
        l_skel = l_sens = None
        if args.modality in ("multimodal", "skeleton"):
            l_skel = ldam_skel(p_skel, y_skel)
        if args.modality in ("multimodal", "sensor"):
            l_sens = ldam_sens(p_sens, y_sens)

        if args.modality == "multimodal":
            loss = (l_skel + l_sens) / 2
        elif args.modality == "skeleton":
            loss = l_skel
        else:
            loss = l_sens
        info_skel = info_sens = {}

    elif args.wm == "gcl":
        l_skel = l_sens = None
        if args.modality in ("multimodal", "skeleton"):
            l_skel = gcl_skel(p_skel, y_skel)
        if args.modality in ("multimodal", "sensor"):
            l_sens = gcl_sens(p_sens, y_sens)

        if consistency is not None:
            lam = args.consistency_lambda
            l_skel += 0.5 * lam * consistency
            l_sens += 0.5 * lam * consistency

        if args.modality == "multimodal":
            loss = (l_skel + l_sens) / 2
        elif args.modality == "skeleton":
            loss = l_skel
        else:
            loss = l_sens

        info_skel = info_sens = {}
    else:
        if args.modality == "multimodal":
            l_skel, info_skel = weighted_branch_loss(p_skel, y_skel, args.wm, sk_counts, train)
            l_sens, info_sens = weighted_branch_loss(p_sens, y_sens, args.wm, se_counts, train)
            loss = (l_skel + l_sens) / 2
        elif args.modality == "skeleton":
            loss, info_skel = weighted_branch_loss(p_skel, y_skel, args.wm, sk_counts, train)
            info_sens = {}
        else:
            loss, info_sens = weighted_branch_loss(p_sens, y_sens, args.wm, se_counts, train)
            info_skel = {}

    if train:
        optimizer.zero_grad()
        if args.modality == "multimodal" and grad_method is not None:
            grad_method.backward(losses=[l_skel, l_sens], shared_parameters=model.get_shared_parameters())
        else:
            loss.backward()
        optimizer.step()

    with torch.no_grad():
        cs = (p_skel.argmax(1) == y_skel).sum().item() if p_skel is not None else 0
        ce = (p_sens.argmax(1) == y_sens).sum().item() if p_sens is not None else 0

    if args.modality == "skeleton":
        n = y_skel.size(0)
    elif args.modality == "sensor":
        n = y_sens.size(0)
    else:
        n = y_skel.size(0)
    return loss.item(), cs, ce, n


# ---------------------------------------------------------------------------
# Epoch Loop
# ---------------------------------------------------------------------------
def run_epoch(
    loader,
    model,
    optimizer,
    args,
    sk_counts,
    se_counts,
    ldam_skel,
    ldam_sens,
    gcl_skel,
    gcl_sens,
    grad_method,
    device,
    train: bool,
    collect_preds: bool = False,
):
    """Run all batches for one train or eval epoch.

    When collect_preds=True, this also returns labels/predictions used for
    fold-level reports and confusion matrices.
    """
    total_loss = total_sk = total_se = total_n = 0
    trues_skel, preds_skel, trues_sens, preds_sens, trues_ens, preds_ens = [], [], [], [], [], []

    for idx, batch in enumerate(loader, start=1):
        l, cs, ce, n = process_batch(
            batch,
            model,
            optimizer,
            args,
            sk_counts,
            se_counts,
            ldam_skel,
            ldam_sens,
            gcl_skel,
            gcl_sens,
            grad_method,
            device,
            train,
        )
        total_loss += l
        total_sk   += cs
        total_se   += ce
        total_n    += n

        if collect_preds and not train:
            with torch.no_grad():
                if args.modality == "multimodal":
                    sk_logits, se_logits = model(
                        flatten_skel(batch["skeleton"].to(device)).float(),
                        batch["sensor"].to(device).float()
                    )
                elif args.modality == "skeleton":
                    sk_logits = model(
                        flatten_skel(batch["skeleton"].to(device)).float()
                    )
                    se_logits = None
                else:
                    sk_logits = None
                    se_logits = model(batch["sensor"].to(device).float())

            if args.modality in ("multimodal", "skeleton"):
                trues_skel.extend(batch["label_skeleton"].tolist())
                preds_skel.extend(sk_logits.argmax(1).cpu().tolist())
            if args.modality in ("multimodal", "sensor"):
                trues_sens.extend(batch["label_sensor"].tolist())
                preds_sens.extend(se_logits.argmax(1).cpu().tolist())

            if args.modality == "multimodal" and args.synchronized_loading:
                ps = F.softmax(sk_logits, dim=1)
                pt = F.softmax(se_logits, dim=1)
                p_ens = ((ps + pt) / 2).argmax(1).cpu().tolist()
                trues_ens.extend(batch["label_skeleton"].tolist())
                preds_ens.extend(p_ens)

        if idx % 10 == 0 or idx == len(loader):
            tag = "Train" if train else " Eval "
            print(f"{tag}-Batch [{idx}/{len(loader)}]: "
                  f"loss={l:.4f}, "
                  f"skel_acc={cs/n*100:.1f}%, "
                  f"sens_acc={ce/n*100:.1f}%")

    avg_loss = total_loss / len(loader)
    acc_skel = total_sk   / total_n * 100
    acc_sens = total_se   / total_n * 100
    
    if collect_preds:
        if args.modality=="multimodal" and args.synchronized_loading:
            ens_acc = sum(p==t for p,t in zip(preds_ens, trues_ens)) / len(trues_ens) * 100
        else:
            ens_acc = None
        return avg_loss, acc_skel, acc_sens, trues_skel, preds_skel, trues_sens, preds_sens, trues_ens, preds_ens, ens_acc
    else:
        return avg_loss, acc_skel, acc_sens


# ---------------------------------------------------------------------------
# Fold Loop
# ---------------------------------------------------------------------------
def train_one_fold(fold_idx, reader, args, train_subj, eval_subj):
    """Train and evaluate one subject-level fold."""
    args.dataset = normalize_dataset_name(args.dataset)
    params = dict(FBG_FOG_PARAMS[args.dataset])
    params.update(getattr(args, "_param_overrides", {}))

    # Data: cached raw readers are converted into train/eval PyTorch loaders.
    train_loader, eval_loader = create_fusion_loaders(
        args.dataset, reader, train_subj, eval_subj,
        batch_size   = params["batch_size"],
        synchronized = args.synchronized_loading,
        seed         = args.seed,
        num_workers  = 4,
        pad_skel     = params["pose_length"],
        pad_sens     = params["sensor_length"],
        modality     = args.modality,
        sync_train_oversample = args.sync_train_oversample)

    # Model/optimizer: model type is selected from --modality.
    model = choose_model(args, params, DEVICE)
    print(f"Total params: {count_params(model):,}")
    optimizer = optim.SGD(
        model.parameters(),
        lr=params["learning_rate"],
        momentum=args.sgd_momentum,
        weight_decay=args.weight_decay,
    )
    grad_method = (CAGrad(n_tasks=2, device=DEVICE, c=args.alpha, max_norm=args.max_norm) if args.alpha > 0 else None)

    # Losses: LDAM/GCL/class weights need branch-specific class counts.
    sk_counts, se_counts = get_branch_class_counts(train_loader, params["num_classes"])
    ldam_skel, ldam_sens, gcl_skel, gcl_sens, drw_weights = build_branch_losses(
        args, sk_counts, se_counts, DEVICE
    )

    best_avg     = 0.0
    best_trues_sk, best_preds_sk, best_trues_se, best_preds_se, best_trues_ens, best_preds_ens  = [], [], [], [], [], []
    best_state = None
    no_improve    = 0
    patience      = args.patience
    train_losses = []
    val_losses   = []
    for ep in range(params["epochs"]): 
        apply_gcl_drw(args, ep, fold_idx, gcl_skel, gcl_sens, drw_weights)

        print(f"\n--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} TRAIN ---")
        model.train()
        tl, tsk, tse = run_epoch(
            train_loader,
            model,
            optimizer,
            args,
            sk_counts,
            se_counts,
            ldam_skel,
            ldam_sens,
            gcl_skel,
            gcl_sens,
            grad_method,
            DEVICE,
            train=True,
        )

        print(f"--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} EVAL  ---")
        model.eval()
        with torch.no_grad():
            vl, vsk, vse, t_sk, p_sk, t_se, p_se, t_ens, p_ens, ens_acc = run_epoch(
                eval_loader,
                model,
                optimizer,
                args,
                sk_counts,
                se_counts,
                ldam_skel,
                ldam_sens,
                gcl_skel,
                gcl_sens,
                grad_method,
                DEVICE,
                train=False,
                collect_preds=True,
            )

        train_losses.append(tl)
        val_losses.append(vl)

        if args.modality=="multimodal" and args.synchronized_loading:
            avg = ens_acc
        else:
            avg = (vsk + vse)/2 if args.modality=="multimodal" else (vsk if args.modality=="skeleton" else vse)

        if avg > best_avg:
            best_avg, best_trues_sk, best_preds_sk, best_trues_se, best_preds_se = avg, t_sk, p_sk, t_se, p_se
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if args.modality=="multimodal":
                best_trues_ens, best_preds_ens = t_ens, p_ens
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Fold {fold_idx}] No improvement for {patience} epochs → early stopping at epoch {ep+1}")
                break

        if args.modality=="multimodal" and args.synchronized_loading:
            print(f"[Fold {fold_idx}][Ep {ep+1}/{params['epochs']}] "
                  f"Train loss={tl:.3f}   acc={tsk:.1f}% | "
                  f"Eval loss={vl:.3f}   ens_acc={avg:.1f}%")
        else:
            print(f"[Fold {fold_idx}][Ep {ep+1}/{params['epochs']}] "
                  f"Train loss={tl:.3f} skel={tsk:.1f}% sen={tse:.1f}% | "
                  f"Eval loss={vl:.3f} skel={vsk:.1f}% sen={vse:.1f}% avg={avg:.1f}%")

    save_loss_curve(args, fold_idx, train_losses, val_losses)

    if args.modality in ("multimodal", "skeleton"):
        sk_acc = sum(1 for i,p in enumerate(best_preds_sk) if p == best_trues_sk[i]) / len(best_trues_sk) * 100
    else:
        sk_acc = 0.0

    if args.modality in ("multimodal", "sensor"):
        se_acc = sum(1 for i,p in enumerate(best_preds_se) if p == best_trues_se[i]) / len(best_trues_se) * 100
    else:
        se_acc = 0.0

    if args.modality=="multimodal" and args.synchronized_loading:
        print(f"\n*** Fold {fold_idx} Best Ensemble Acc: {best_avg:.2f}% ***\n")
    else:
        print(f"\n*** Fold {fold_idx} Best skel={sk_acc:.2f}%  sens={se_acc:.2f}%, avg={best_avg:.2f}% ***\n")

    if args.modality in ("multimodal", "skeleton"):
        if args.modality == "multimodal" and args.synchronized_loading:
            print("Skeleton Report at Best Ensemble Epoch:")
        else:
            print("Best Skeleton Report:")
        print(classification_report(best_trues_sk, best_preds_sk, digits=2, zero_division=0))
        if args.modality == "multimodal" and args.synchronized_loading:
            print("Skeleton Confusion Matrix at Best Ensemble Epoch:")
        else:
            print("Best Skeleton Confusion Matrix:")
        print(confusion_matrix(best_trues_sk, best_preds_sk))

    if args.modality in ("multimodal", "sensor"):
        if args.modality == "multimodal" and args.synchronized_loading:
            print("Sensor Report at Best Ensemble Epoch:")
        else:
            print("Best Sensor Report:")
        print(classification_report(best_trues_se, best_preds_se, digits=2, zero_division=0))
        if args.modality == "multimodal" and args.synchronized_loading:
            print("Sensor Confusion Matrix at Best Ensemble Epoch:")
        else:
            print("Best Sensor Confusion Matrix:")
        print(confusion_matrix(best_trues_se, best_preds_se))

    if (
        args.eval_missing_ratios
        and args.modality == "multimodal"
        and args.synchronized_loading
        and best_state is not None
    ):
        model.load_state_dict(best_state, strict=True)

        def predict_logits(skel, sens, use_skel, use_sens):
            log_skel, log_sens = model(skel, sens)
            probs = []
            if use_skel:
                probs.append(F.softmax(log_skel, dim=1))
            if use_sens:
                probs.append(F.softmax(log_sens, dim=1))
            return sum(probs) / len(probs)

        for available, missing, use_skel, use_sens in TWO_MODALITY_SUBSETS:
            score = eval_two_modality_subset_sync(
                model, eval_loader, DEVICE, flatten_skel, predict_logits, use_skel, use_sens
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
                    model, eval_loader, ratio, ratio_seed, DEVICE, flatten_skel, predict_logits
                )
                repeat_scores.append(float(score))
            repeat_arr = np.array(repeat_scores, dtype=float)
            args._missing_ratio_scores[ratio].append(float(repeat_arr.mean()))
            print(
                f"[SYNC][missing_ratio={ratio:.2f}] "
                f"fold_mean={repeat_arr.mean():5.2f}% ± {repeat_arr.std():4.2f}% "
                f"over {args.missing_ratio_repeats} repeats"
            )
        
    if args.modality=="multimodal" and args.synchronized_loading:
        print("Best Ensemble Report:")
        print(classification_report(best_trues_ens, best_preds_ens, digits=2))
    return sk_acc, se_acc, best_avg


# ---------------------------------------------------------------------------
# Cross-Validation Driver
# ---------------------------------------------------------------------------
def main(args):
    set_random_seed(args.seed)
    args.dataset = normalize_dataset_name(args.dataset)
    apply_param_preset(args)
    apply_train_param_overrides(args)
    if args.wm.lower() == "gcl" and not args.use_norm_and_cos and not args.legacy_gcl_head:
        args.use_norm_and_cos = True
        print("[INFO] Enabled use_norm_and_cos because GCL requires bounded cosine logits.")
    elif args.wm.lower() == "gcl" and args.legacy_gcl_head:
        print("[WARN] Using legacy GCL head without forced normalization/cosine logits.")
    print("Effective arguments: ", args)
    args._missing_ratios = parse_ratio_list(args.missing_ratios)

    reader = load_reader(args.dataset, rebuild=args.rebuild_cache)
    folds = generate_class_stratified_folds(reader, args.dataset)

    if args.modality=="all":
        modes = ["skeleton", "sensor", "multimodal"]
    elif args.modality=="both":
        modes = ["skeleton", "sensor"]
    else:
        modes = [args.modality]

    for mod in modes:
        args.modality = mod
        if args.eval_missing_ratios and args.synchronized_loading and mod == "multimodal":
            args._missing_ratio_scores = {ratio: [] for ratio in args._missing_ratios}
            args._subset_scores = {available: [] for available, *_ in TWO_MODALITY_SUBSETS}
        print(f"\n>>> MODE: {mod.upper()} <<<")
        results = []
        for idx,(t,e) in enumerate(folds,1):
            print(f"\nFold {idx}: train={t}, eval={e}")
            results.append(train_one_fold(idx, reader, args, t, e))
        arr = np.array(results)
        mean_sk, mean_se, mean_av = arr.mean(axis=0)
        if args.modality=="multimodal" and args.synchronized_loading:
            print(f"mean Ensemble Acc: {mean_av:.2f}%")
            if args.eval_missing_ratios and getattr(args, "_subset_scores", None) and all(args._subset_scores.values()):
                print("\n=== Modality-subset accuracy at best epoch (sync deterministic mask) ===")
                print("Policy: deterministic subset evaluation; missing modalities are zero-filled.")
                for available, missing, *_ in TWO_MODALITY_SUBSETS:
                    subset_arr = np.array(args._subset_scores[available], dtype=float)
                    print(
                        f"[available={available}][missing={missing}] "
                        f"{subset_arr.mean():5.2f}% ± {subset_arr.std():4.2f}%  over {len(subset_arr)} folds"
                    )
            if args.eval_missing_ratios and args._missing_ratio_scores and all(args._missing_ratio_scores.values()):
                print("\n=== Missing-ratio accuracy at best epoch (sync inference stress test) ===")
                print(f"Policy: sample-level Method B, {args.missing_ratio_repeats} repeats per fold; selected samples miss one modality.")
                for ratio in args._missing_ratios:
                    ratio_arr = np.array(args._missing_ratio_scores[ratio], dtype=float)
                    print(f"[ratio={ratio:.2f}] {ratio_arr.mean():5.2f}% ± {ratio_arr.std():4.2f}%  over {len(ratio_arr)} folds")
        else:
            print(f"mean skel={mean_sk:.2f}%, sensor={mean_se:.2f}%, avg={mean_av:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FBG/FoG skeleton, sensor, or multimodal multitask models."
    )
    parser.add_argument("--dataset", type=str, default='fog', choices=["fbg", "fog"], help="Dataset to use")
    parser.add_argument("--modality", type=str, default='multimodal', choices=["skeleton", "sensor", "both", "multimodal", "all"])
    parser.add_argument("--consistency_lambda", type=float, default=1, help="weight on symmetric KL consistency loss (sync+multimodal only)")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--wm", type=str, default='gcl', choices=["ce", "class_wt", "ldam", "gcl"], help="Weighting method")
    parser.add_argument("--synchronized_loading", action="store_true", help="time-align pose+sensor for both train & eval")
    parser.add_argument("--sync_train_oversample", action="store_true",
                        help="Oversample synchronized training pairs; default keeps original train pair distribution.")
    parser.add_argument("--alpha", type=float, default=0.3, help="CAGrad c parameter")
    parser.add_argument("--max_norm", type=float, default=0.0, help="CAGrad gradient clipping norm; 0 disables clipping")
    parser.add_argument("--sgd_momentum", type=float, default=0.9,
                        help="SGD momentum.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="SGD weight decay.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override dataset/preset training epochs.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override dataset/preset training batch size.")
    parser.add_argument("--patience", type=int, default=100,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--ldam_s", type=float, default=30)
    parser.add_argument("--ldam_m", type=float, default=0.5)
    parser.add_argument("--gcl_m", type=float, default=0.4)
    parser.add_argument("--gcl_s", type=float, default=40)
    parser.add_argument("--noise_mul", type=float, default=0.0)
    parser.add_argument("--drw_warmup", type=int, default=5, help="Number of warmup epochs before applying class re-weighting (DRW) for GCL")
    parser.add_argument("--use_norm_and_cos", action="store_true")
    parser.add_argument("--legacy_gcl_head", action="store_true",
                        help="Use the pre-normalized-head GCL setup for reproducing old FoG runs.")
    parser.add_argument("--param_preset", type=str, default="",
                        help=f"Named FBG/FoG parameter preset. Available: {', '.join(sorted(FBG_FOG_PARAM_PRESETS))}")
    parser.add_argument("--save_loss_plots", action="store_true", help="Save train/eval loss curves per fold")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild the dataset .pkl cache before training")
    parser.add_argument("--eval_missing_ratios", action="store_true",
                        help="Run sync-only sample-level inference-time missing-ratio evaluation after the best epoch.")
    parser.add_argument("--missing_ratios", type=str, default="0.2,0.4,0.6,0.8",
                        help="Comma-separated missing ratios for --eval_missing_ratios.")
    parser.add_argument("--missing_ratio_repeats", type=int, default=5,
                        help="Random repeats per fold for --eval_missing_ratios.")
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    print("Arguments: ", parsed_args)
    main(parsed_args)
