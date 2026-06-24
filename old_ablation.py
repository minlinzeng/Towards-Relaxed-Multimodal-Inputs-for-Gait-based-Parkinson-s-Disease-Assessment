import os
import torch
import math
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from data.public_pd_datareader import PDReader
from feature_encoder import MultiModalMultiTaskModel, SkelModalityModel, SensorModalityModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optimizers.weight_methods import CAGrad
from optimizers.losses import GCLLoss
from processPdfeData import pdfeReader
from sklearn.metrics import classification_report
from data.efficient_dataloader import create_fusion_loaders
from sklearn.metrics import classification_report, confusion_matrix
from utility import count_params
import matplotlib
matplotlib.use('Agg')       # use non-interactive backend on servers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATHS = {
    "walk": {
        "pose_path": "./PD_3D_motion-capture_data/C3Dfiles_processed_new",        
        # "pose_path": "./PD_3D_motion-capture_data/C3Dfiles_cleaned_sequences",
        "sensor_path": "./PD_3D_motion-capture_data/GRF_processed",
        "label_path": "./PD_3D_motion-capture_data/PDGinfo.xlsx",
    },
    "turn": {
        "pose_path":    './PD_3D_motion-capture_data/turn-in-place/predictions/',
        "sensor_path":  './PD_3D_motion-capture_data/turn-in-place/IMU/',
        "label_path":   './PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx',
        "lifted_path":  './PD_3D_motion-capture_data/turn-in-place/lifted/',
    }
}

MODALITY_PARAMS = {
    "walk": {
        "pose_length": 101,
        "skeleton_input_dim": 51,
        "skeleton_output_dim": 3,
        "sensor_in_channels": 3,
        "sensor_out_channels": 3,
        "sensor_length": 65,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "taskhead_input_dim": 8*16,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 256,
        # "class_counts": [18, 14, 11]
    },
    "turn": {
        "pose_length": 101,
        "skeleton_input_dim": 21,
        "skeleton_output_dim": 6,
        "sensor_in_channels": 6,
        "sensor_out_channels": 6,
        "sensor_length": 426,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "taskhead_input_dim": 8*16,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 256,
        # "class_counts": [6, 25, 3]
    }
}

MODEL_KEYS = ("skeleton_input_dim", "skeleton_output_dim", "sensor_in_channels", \
              "sensor_out_channels", "sensor_length", "shared_out_channels", "backbone_dim", "taskhead_input_dim", "num_classes")

# ─── Utility Functions ─────
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
        B, F, J, C = x.shape
        return x.view(B, F, J * C)
    return x

# ─── Model Selection ─────────────
def choose_model(args, params):
    if args.modality == 'skeleton':
        model = SkelModalityModel(
            skeleton_input_dim   = params["skeleton_input_dim"],
            skeleton_output_dim  = params["skeleton_output_dim"],
            sensor_out_channels  = params["skeleton_output_dim"],  
            shared_out_channels  = params["shared_out_channels"],
            backbone_dim         = params["backbone_dim"],
            taskhead_input_dim   = params["taskhead_input_dim"],
            num_classes          = params["num_classes"],
        ).to(device)
    elif args.modality == 'sensor':
        model = SensorModalityModel(
            sensor_in_channels   = params["sensor_in_channels"],
            sensor_out_channels  = params["sensor_out_channels"],
            sensor_length        = params["sensor_length"],
            shared_out_channels  = params["shared_out_channels"],
            backbone_dim         = params["backbone_dim"],
            taskhead_input_dim   = params["taskhead_input_dim"],
            num_classes          = params["num_classes"],
        ).to(device)
    else:
        model = MultiModalMultiTaskModel(
            skeleton_input_dim   = params["skeleton_input_dim"],
            skeleton_output_dim  = params["skeleton_output_dim"],
            sensor_in_channels   = params["sensor_in_channels"],
            sensor_out_channels  = params["sensor_out_channels"],
            sensor_length        = params["sensor_length"],
            shared_out_channels  = params["shared_out_channels"],
            backbone_dim         = params["backbone_dim"],
            taskhead_input_dim   = params["taskhead_input_dim"],
            num_classes          = params["num_classes"],
            use_norm             = args.use_norm_and_cos,
            use_cosine           = args.use_norm_and_cos,
            synchronized_loading = args.synchronized_loading,
        ).to(device)
        
    
    return model

# ─── Training Helpers ────────
def get_branch_class_counts(loader, num_classes):
    """
    Returns two lists of length num_classes:
      - skeleton_counts[j] = total # of samples with label_skeleton == j
      - sensor_counts[j]   = total # of samples with label_sensor   == j
    """
    sk_counter = Counter()
    se_counter = Counter()
    for batch in loader:
        # extract both label tensors
        if "label_skeleton" in batch:
            labels_sk = batch["label_skeleton"].cpu().tolist()
            sk_counter.update(labels_sk)
        if "label_sensor" in batch:
            labels_se = batch["label_sensor"].cpu().tolist()
            se_counter.update(labels_se)

    # turn into fixed‐length lists
    skeleton_counts = [sk_counter[i] for i in range(num_classes)]
    sensor_counts   = [se_counter[i] for i in range(num_classes)]
    
    print(f"Skeleton counts: {skeleton_counts}, Sensor counts: {sensor_counts}")
    return skeleton_counts, sensor_counts

def generate_class_stratified_folds(reader, dataset):
    """
    - walk: eval folds only from subjects with both modalities;
            train = the rest (still filtered later by modality if needed)
    - turn: unchanged
    """
    from collections import defaultdict
    import random

    if dataset == "walk":
        # 1) which prefixes truly have both pose & sensor?
        pose_pfx = { "_".join(k.split("_")[:2]) for k in reader.pose_dict.keys() }
        sens_pfx = { "_".join(k.split("_")[:2]) for k in reader.sensor_dict.keys() }
        both_modal = pose_pfx & sens_pfx

        # 2) raw per-subject labels, but only keep both-modal
        raw = reader.pose_label_dict
        label_dict = { s: raw[s] for s in raw if s in both_modal }

    elif dataset == "turn":
        label_dict = {
            s: reader.labels_dict[s][0]
            for s in reader.labels_dict
            if s not in ("SUB10","SUB30","SUB22")
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 3) group by class
    class_to_subj = defaultdict(list)
    for subj, lbl in label_dict.items():
        class_to_subj[lbl].append(subj)

    # 4) number of folds = size of smallest class
    m = min(len(lst) for lst in class_to_subj.values())
    if m == 0:
        raise ValueError("Need ≥1 subject/class with both modalities for walk")

    # 5) balance each class down/up to m
    balanced = {}
    for lbl, subs in class_to_subj.items():
        if len(subs) > m:
            subs = random.sample(subs, k=m)
        random.shuffle(subs)
        balanced[lbl] = subs

    # 6) build folds
    folds = []
    for i in range(m):
        eval_subj = [balanced[lbl][i] for lbl in sorted(balanced)]
        train_subj = [s for s in label_dict if s not in eval_subj]
        folds.append((train_subj, eval_subj))

    return folds

# ─── Loss Helpers ─────
def make_inv_freq_weights(counts):
    w = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    w = w / w.sum() * len(counts)
    return w

def compute_log_based_weights(counts, div, device):
    """
    Given a list/array of class counts, return a torch.Tensor of
    log-based weights, i.e.:
      raw_w[j] = log(max_count / counts[j] + 0.01) / div
    then clipped to >=0 and re-normalized so sum(raw_w)==num_classes.
    """
    counts_np = np.array(counts, dtype=np.float32)
    max_count = counts_np.max()
    # 1) raw log-based
    raw = np.log(max_count / counts_np + 0.01) / div
    # 2) clamp to zero (in case of extremely large ratios)
    raw = np.clip(raw, a_min=0.0, a_max=None)
    # 3) normalize so sum(weights) == num_classes
    if raw.sum() > 0:
        raw = raw / raw.sum() * len(raw)
    # 4) to torch.Tensor on correct device
    return torch.tensor(raw, dtype=torch.float32, device=device)

def weighted_branch_loss(logits, labels, method, class_counts, train=True):
    C = logits.shape[1]
    if method == "ce":
        return nn.CrossEntropyLoss()(logits, labels), {}
    if method == "class_wt":
        w = 1.0 / (torch.tensor(class_counts, device=labels.device) + 1e-8)
        w = w / w.sum() * C
        return nn.CrossEntropyLoss(weight=w)(logits, labels), {"weights": w}
    raise ValueError(method)

# ─── Core Training Loop ───────────
def process_batch(batch, model, optimizer, args, sk_counts, se_counts, gcl_skel, gcl_sens, grad_method, device, train: bool):
    """Forward, compute losses, backward (if train), return (loss, correct_skel, correct_sens, n)."""
    # 1) unpack inputs
    skeleton, sensor = None, None
    if "skeleton" in batch:
        skeleton = flatten_skel(batch["skeleton"].to(device)).float()
        y_skel   = batch["label_skeleton"].to(device).long()
    if "sensor" in batch:
        sensor   = batch["sensor"].to(device).float()
        y_sens   = batch["label_sensor"].to(device).long()

    # 2) forward + two‐head losses
    if args.modality == "multimodal":
        p_skel, p_sens = model(skeleton, sensor)
        
        # 2.5) consistency regularization (sync+multimodal only)
        consistency = None
        if args.synchronized_loading:
            # convert to distributions
            logp = F.log_softmax(p_skel, dim=1)
            q    = F.softmax(p_sens, dim=1)
            kl1  = F.kl_div(logp, q, reduction="batchmean")

            logq = F.log_softmax(p_sens, dim=1)
            p    = F.softmax(p_skel, dim=1)
            kl2  = F.kl_div(logq, p, reduction="batchmean")
            consistency = kl1 + kl2      
    elif args.modality == "skeleton":
        p_skel, p_sens = model(skeleton), None
    elif args.modality == "sensor":  # sensor‐only
        p_skel, p_sens = None, model(sensor)

    # 3) loss computation
    if args.wm == "gcl":
        if gcl_skel is not None:
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
            if args.modality in ("multimodal", "skeleton"):
                l_skel, info_skel = weighted_branch_loss(p_skel, y_skel, "ce", sk_counts, train)
            else:
                l_skel = None; info_skel = {}
            if args.modality in ("multimodal", "sensor"):
                l_sens, info_sens = weighted_branch_loss(p_sens, y_sens, "ce", se_counts, train)
            else:
                l_sens = None; info_sens = {}

            if args.modality == "multimodal":
                loss = (l_skel + l_sens) / 2
            elif args.modality == "skeleton":
                loss = l_skel
            else:
                loss = l_sens
    else: # CE, class‐wt
        if args.modality == "multimodal":
            l_skel, info_skel = weighted_branch_loss(p_skel, y_skel, args.wm, sk_counts, train)
            l_sens, info_sens = weighted_branch_loss(p_sens, y_sens, args.wm, se_counts, train)            
            loss = (l_skel + l_sens) / 2
        elif args.modality == "skeleton":
            loss, info_skel = weighted_branch_loss(p_skel, y_skel, args.wm, sk_counts, train)
            info_sens = {}
        else:  # sensor‐only
            loss, info_sens = weighted_branch_loss(p_sens, y_sens, args.wm, se_counts, train)
            info_skel = {}

    # 3) backward step
    if train:
        optimizer.zero_grad()
        if args.modality == "multimodal" and grad_method is not None:
            grad_method.backward(losses=[l_skel, l_sens], shared_parameters=model.get_shared_parameters())
        else:
            loss.backward()
        optimizer.step()

    # 4) compute accuracy
    with torch.no_grad():
        cs = (p_skel.argmax(1) == y_skel).sum().item() if p_skel is not None else 0
        ce = (p_sens.argmax(1) == y_sens).sum().item() if p_sens is not None else 0

    if args.modality == "skeleton":
        n = y_skel.size(0)
    elif args.modality == "sensor":
        n = y_sens.size(0)
    else:  # multimodal
        n = y_skel.size(0)
    return loss.item(), cs, ce, n

def run_epoch(loader, model, optimizer, args, sk_counts, se_counts, gcl_skel, gcl_sens, grad_method, device, train: bool, collect_preds: bool = False):
    """Loop over one epoch, return stats and optionally predictions for confusion matrices."""
    total_loss = total_sk = total_se = total_n = 0
    trues_skel, preds_skel, trues_sens, preds_sens, trues_ens, preds_ens = [], [], [], [], [], []

    for idx, batch in enumerate(loader, start=1):
        l, cs, ce, n = process_batch(batch, model, optimizer, args, sk_counts, se_counts, gcl_skel, gcl_sens, grad_method, device, train)
        total_loss += l
        total_sk   += cs
        total_se   += ce
        total_n    += n

        # collect logits+labels for eval
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
                else:  # sensor-only
                    sk_logits = None
                    se_logits = model(batch["sensor"].to(device).float())

            # append predictions and truths
            if args.modality in ("multimodal", "skeleton"):
                trues_skel.extend(batch["label_skeleton"].tolist())
                preds_skel.extend(sk_logits.argmax(1).cpu().tolist())
            if args.modality in ("multimodal", "sensor"):
                trues_sens.extend(batch["label_sensor"].tolist())
                preds_sens.extend(se_logits.argmax(1).cpu().tolist())
                
            # For multimodal, also do ensembled prediction
            if args.modality == "multimodal" and args.synchronized_loading:
                # turn logits into probs, average, then argmax
                ps = F.softmax(sk_logits, dim=1)
                pt = F.softmax(se_logits, dim=1)
                p_ens = ((ps + pt) / 2).argmax(1).cpu().tolist()
                trues_ens.extend(batch["label_skeleton"].tolist())  # same label
                preds_ens.extend(p_ens)

        # optional logging
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

def train_one_fold(fold_idx, reader, args, train_subj, eval_subj):
    # 1) data loaders
    train_loader, eval_loader = create_fusion_loaders(
        args.dataset, reader, train_subj, eval_subj,
        batch_size   = MODALITY_PARAMS[args.dataset]["batch_size"],
        synchronized = args.synchronized_loading,
        seed         = args.seed,
        num_workers  = 4,
        pad_skel     = MODALITY_PARAMS[args.dataset]["pose_length"],
        pad_sens     = MODALITY_PARAMS[args.dataset]["sensor_length"],
        modality     = args.modality)
    params = MODALITY_PARAMS[args.dataset]
    
    # 2) model & optimizers
    model = choose_model(args, params)
    total = count_params(model)                 # trainable params
    print(f"Total params: {total:,}")
    # if args.wm == 'gcl' and args.gcl_m > 0:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=1e-4)
    # else:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=1e-4)
    grad_method = (CAGrad(n_tasks=2, device=device, c=args.alpha) if args.alpha>0 else None)
    
    # 3) Precompute class counts for weighted loss
    sk_counts, se_counts = get_branch_class_counts(train_loader, params["num_classes"])
    if args.wm.lower() == "gcl" and args.gcl_m > 0:
        w_skel = make_inv_freq_weights(sk_counts)
        w_sens = make_inv_freq_weights(se_counts)
        gcl_skel = GCLLoss(cls_num_list=sk_counts, m=args.gcl_m, s=args.gcl_s, noise_mul=args.noise_mul, weight=None).to(device)
        gcl_sens = GCLLoss(cls_num_list=se_counts, m=args.gcl_m, s=args.gcl_s, noise_mul=args.noise_mul, weight=None).to(device)
    else:
        gcl_skel = gcl_sens = None
    
    # 4) Training loop
    best_avg     = 0.0
    best_trues_sk, best_preds_sk, best_trues_se, best_preds_se, best_trues_ens, best_preds_ens  = [], [], [], [], [], []
    no_improve    = 0
    patience      = 100
    train_losses = []
    val_losses   = []
    for ep in range(params["epochs"]): 
        if args.wm.lower() == "gcl" and args.gcl_m > 0 and ep == args.drw_warmup:
            print(f"[Fold {fold_idx}] DRW: applying class weights at epoch {ep+1}")
            # assign the precomputed weight tensors
            gcl_skel.weight = w_skel
            gcl_sens.weight = w_sens

        print(f"\n--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} TRAIN ---")
        model.train()
        tl, tsk, tse = run_epoch(train_loader, model, optimizer, args, sk_counts, se_counts, gcl_skel, gcl_sens, grad_method, device, train=True)

        print(f"--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} EVAL  ---")
        model.eval()
        with torch.no_grad():
            vl, vsk, vse, t_sk, p_sk, t_se, p_se, t_ens, p_ens, ens_acc = run_epoch(eval_loader, model, optimizer, args, sk_counts, se_counts, gcl_skel, gcl_sens, grad_method, device, train=False, collect_preds=True)

        train_losses.append(tl)
        val_losses.append(vl)
        # scheduler.step(vl)
        if args.modality=="multimodal" and args.synchronized_loading:
            avg = ens_acc
        else:
            avg = (vsk + vse)/2 if args.modality=="multimodal" else (vsk if args.modality=="skeleton" else vse)
        if avg > best_avg:
            best_avg, best_trues_sk, best_preds_sk, best_trues_se, best_preds_se = avg, t_sk, p_sk, t_se, p_se
            # ensemble only in multimodal
            if args.modality=="multimodal":
                best_trues_ens, best_preds_ens = t_ens, p_ens
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Fold {fold_idx}] No improvement for {patience} epochs → early stopping at epoch {ep+1}")
                break
        # print(f"[Fold {fold_idx}][Ep {ep+1}] Train loss={tl:.3f} skel={tsk:.1f}% sen={tse:.1f}% | Val loss={vl:.3f} skel={vsk:.1f}% sen={vse:.1f}% avg={avg:.1f}%")
        if args.modality=="multimodal" and args.synchronized_loading: # sync‐multimodal: only show the ensemble acc
            print(f"[Fold {fold_idx}][Ep {ep+1}/{params['epochs']}] "
                  f"Train loss={tl:.3f}   acc={tsk:.1f}% | "
                  f"Eval loss={vl:.3f}   ens_acc={avg:.1f}%")
        else: # all other modes: show per-head & average
            print(f"[Fold {fold_idx}][Ep {ep+1}/{params['epochs']}] "
                  f"Train loss={tl:.3f} skel={tsk:.1f}% sen={tse:.1f}% | "
                  f"Eval loss={vl:.3f} skel={vsk:.1f}% sen={vse:.1f}% avg={avg:.1f}%")

    epochs = list(range(1, len(train_losses)+1))

    # ——— final best metrics ———
    if args.modality in ("multimodal", "skeleton"):
        sk_acc = sum(1 for i,p in enumerate(best_preds_sk) if p == best_trues_sk[i]) / len(best_trues_sk) * 100
    else:
        sk_acc = 0.0

    # sensor accuracy only if we ran sensor or multimodal
    if args.modality in ("multimodal", "sensor"):
        se_acc = sum(1 for i,p in enumerate(best_preds_se) if p == best_trues_se[i]) / len(best_trues_se) * 100
    else:
        se_acc = 0.0

    if args.modality=="multimodal" and args.synchronized_loading:
        print(f"\n*** Fold {fold_idx} Best Ensemble Acc: {best_avg:.2f}% ***\n")
    else:
        print(f"\n*** Fold {fold_idx} Best skel={sk_acc:.2f}%  sens={se_acc:.2f}%, avg={best_avg:.2f}% ***\n")

    # ——— classification reports ———
    if args.modality in ("multimodal", "skeleton"):
        print("Best Skeleton Report:")
        print(classification_report(best_trues_sk, best_preds_sk, digits=2, zero_division=0))
        print("Best Skeleton Confusion Matrix:")
        print(confusion_matrix(best_trues_sk, best_preds_sk))

    if args.modality in ("multimodal", "sensor"):
        print("Best Sensor Report:")
        print(classification_report(best_trues_se, best_preds_se, digits=2, zero_division=0))
        print("Best Sensor Confusion Matrix:")
        print(confusion_matrix(best_trues_se, best_preds_se))
        
    if args.modality=="multimodal" and args.synchronized_loading:
        print("Best Ensemble Report:")
        print(classification_report(best_trues_ens, best_preds_ens, digits=2))
    return sk_acc, se_acc, best_avg


def main(args):
    set_random_seed(args.seed)
    
    # reader & fold split
    if args.dataset=="walk":
        reader = PDReader(joints_path  = PATHS["walk"]["pose_path"], sensor_path= PATHS["walk"]["sensor_path"], labels_path = PATHS["walk"]["label_path"])
    else:
        reader = pdfeReader(pose_path   = PATHS["turn"]["pose_path"], sensor_path = PATHS["turn"]["sensor_path"], label_path  = PATHS["turn"]["label_path"], lifted_path = PATHS["turn"]["lifted_path"])
    folds = folds = generate_class_stratified_folds(reader, args.dataset)

    # choose what modality to train
    if args.modality=="all":
        modes = ["skeleton","sensor","multimodal"] # train each one by one without reloading data
    elif args.modality=="both":
        modes = ["skeleton","sensor"]
    else:
        modes = [args.modality]

    for mod in modes:
        args.modality = mod
        print(f"\n>>> MODE: {mod.upper()} <<<")
        results = []
        for idx,(t,e) in enumerate(folds,1):
            print(f"\nFold {idx}: train={t}, eval={e}")
            results.append(train_one_fold(idx, reader, args, t, e))
        arr = np.array(results)
        mean_sk, mean_se, mean_av = arr.mean(axis=0)
        if args.modality=="multimodal" and args.synchronized_loading:
            print(f"→ mean Ensemble Acc: {mean_av:.2f}%")
        else:
            print(f"→ mean skel={mean_sk:.2f}%, sensor={mean_se:.2f}%, avg={mean_av:.2f}%")    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='turn', help="Dataset to use: walk or turn")
    parser.add_argument("--modality", type=str, default='multimodal', help="skeleton, sensor, both, multimodal, or all") # both is for training with two single modality consecutively 
    parser.add_argument("--consistency_lambda", type=float, default=1, help="weight on symmetric KL consistency loss (sync+multimodal only)")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--wm", type=str, default='gcl', help="Weighting method: ce, class_wt, ldam, or gcl")
    parser.add_argument("--synchronized_loading", action="store_true", help="time-align pose+sensor for both train & eval")
    parser.add_argument("--alpha", type=float, default=0.1, help="CAGrad c parameter (default 0.4)")
    parser.add_argument("--gcl_m", type=float, default=0.2)
    parser.add_argument("--gcl_s", type=float, default=25)
    parser.add_argument("--noise_mul", type=float, default=0)
    parser.add_argument("--drw_warmup", type=int, default=0, help="Number of warmup epochs before applying class re-weighting (DRW) for GCL")
    parser.add_argument("--use_norm_and_cos", action="store_true")
    print("Arguments: ", parser.parse_args()) 
    main(parser.parse_args())

# ablation on two components
"""
# Turn Dataset Ablation Runs
# neither
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --modality skeleton --wm class_wt --alpha 0 --gcl_m 0 \
    --seed "$seed" \
    > "logs/turn/mm_alpha0_1/gcl/async/ablation/none_b256_sgd_ce_seed${seed}.out" 2>&1 &
  sleep 0.2
done

# CAGrad only: 
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --wm ce \
    --alpha 0.1 \
    --seed "$seed" \
    > "logs/turn/mm_alpha0_1/gcl/async/both_imbalance/a0.1_b16_sgd_ce_seed${seed}.out" 2>&1 &
  sleep 0.2
done

for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --wm gcl \
    --gcl_m 0.5 \
    --gcl_s 30 \
    --alpha 0 \
    --noise_mul 0.1 \
    --drw_warmup 10 \
    --use_norm_and_cos \
    --seed "$seed" \
    > "logs/turn/mm_alpha0_1/gcl/async/ablation/gcl_m0.5_nm0.1_s30_wu10_256b_sgd_ce_cosln_seed${seed}.out" 2>&1 &
  sleep 0.2
done

# both CAGrad and GCL
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --wm gcl \
    --gcl_m 0.5 \
    --gcl_s 30 \
    --noise_mul 0.1 \
    --drw_warmup 10 \
    --use_norm_and_cos \
    --seed "$seed" \
    > "logs/turn/mm_alpha0_1/gcl/async/ablation/gcl_m0.5_nm0.1_s30_wu10_256b_sgd_ce_cosln_seed${seed}.out" 2>&1 &
  sleep 0.2
done
"""

"""
# walk dataset ablation runs
# neither
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --dataset walk \
    --wm ce \
    --alpha 0 \
    --seed "$seed" \
    > "logs/walk/ablation/none_b64_sgd_seed${seed}.out" 2>&1 &
  sleep 0.2
done

# CAGrad only: 
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --dataset walk \
    --wm ce \
    --alpha 0.1 \
    --seed "$seed" \
    > "logs/walk/both_imbalance/mm/async/a0.1_b16_sgd_ce_seed${seed}.out" 2>&1 &
  sleep 0.2
done

# GCL only:
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --dataset walk \
    --wm gcl \
    --gcl_m 0.1 \
    --gcl_s 30 \
    --alpha 0 \
    --noise_mul 0.1 \
    --drw_warmup 10 \
    --seed "$seed" \
    > "logs/walk/ablation/gcl_m0.1_nm0.1_s30_wu10_64b_sgd_seed${seed}.out" 2>&1 &
  sleep 0.2
done
"""

"""
kill $(jobs -p)
jobs -p

####### CAGrad
# Define parameter lists
seeds=(42 43 44 2 3 4)
for seed in "${seeds[@]}"; do
    # Replace dot with underscore for safe filenames
    sanitized_alpha=$(echo "$alpha" | sed 's/\./_/g')
    nohup python3 single_modality.py \
        --dataset turn \
        --synchronized_loading \
        --wm class_wt \
        --seed "${seed}" \
        > "logs/turn/mm_alpha0_1/wl/sync/0527_log_1task_s${seed}.out" 2>&1 &
    sleep 0.5  # slight pause to prevent job flooding
done



MAX_PROCS=12
seeds=(44 43 42 2 3 4)
ms=(0.2)
nms=(0)
ss=(20)
wus=(11 15 9 25)
lds=(0 1)

running_jobs() {
    jobs -r | wc -l
}

for m in "${ms[@]}"; do
  for nm in "${nms[@]}"; do
    for s in "${ss[@]}"; do
      for wu in "${wus[@]}"; do
        for ld in "${lds[@]}"; do
          for seed in "${seeds[@]}"; do

            # Wait if too many jobs are already running
            while [ $(pgrep -fc "python3 single_modality.py") -ge $MAX_PROCS ]; do
              sleep 10  # wait and recheck
            done

            echo "Launching: m=$m, nm=$nm, s=$s, wu=$wu, seed=$seed, ld=$ld"
            nohup python3 single_modality.py \
              --dataset turn \
              --synchronized_loading \
              --modality multimodal \
              --wm gcl \
              --gcl_m $m \
              --noise_mul $nm \
              --drw_warmup $wu \
              --gcl_s $s \
              --seed $seed \
              --consistency_lambda $ld \
              > "logs/turn/mm_alpha0_1/gcl/sync/grid/s20_m0.2_nm0/gcl_m${m}_nm${nm}_s${s}_wu${wu}_seed${seed}_ld${ld}.out" 2>&1 &

            sleep 0.5  # slight delay to stabilize launch
          done
        done
      done
    done
  done
done

"""

"""
#!/usr/bin/env bash
# → SAVED AS run_group1.sh (for example)
# This loop will only count its own “--log_based” processes, up to MAX1=12

MAX1=12
seeds=(44 43 42 2 3 4)
cs=(0.1 0.5 1)
ms=(0 0.1 0.2)
nms=(0 0.1 0.2)
ss=(10 20 30)
wus=(10 20 30)
divs=(0.1 1.0 5.0)

for c in "${cs[@]}"; do
  for m in "${ms[@]}"; do
    for nm in "${nms[@]}"; do
      for s in "${ss[@]}"; do
        for wu in "${wus[@]}"; do
          for div in "${divs[@]}"; do
            for seed in "${seeds[@]}"; do

              logfile="logs/turn/mm_alpha0_1/gcl/async/grid/gcl_c${c}_m${m}_nm${nm}_s${s}_wu${wu}_seed${seed}_div${div}.out"

              if [ -f "$logfile" ]; then
                echo "Skipping existing file: $logfile"
                continue
              fi

              # — count only “--log_based” processes:
              count_group1=$( 
                pgrep -af "python3 single_modality.py" \
                  | grep -- "--log_based" \
                  | wc -l 
              )

              # wait until fewer than MAX1 of those exist
              while [ "$count_group1" -ge "$MAX1" ]; do
                sleep 10
                count_group1=$(
                  pgrep -af "python3 single_modality.py" \
                    | grep -- "--log_based" \
                    | wc -l
                )
              done

              echo "Launching (group1): c=$c, m=$m, nm=$nm, s=$s, wu=$wu, seed=$seed, div=$div"
              nohup python3 single_modality.py \
                --dataset turn \
                --alpha "$c" \
                --modality multimodal \
                --wm gcl \
                --gcl_m "$m" \
                --noise_mul "$nm" \
                --drw_warmup "$wu" \
                --gcl_s "$s" \
                --seed "$seed" \
                --log_based \
                --div "$div" \
                > "$logfile" 2>&1 < /dev/null &

              disown
              sleep 0.5

            done
          done
        done
      done
    done
  done
done


"""

"""
MAX_PROCS=28
DATA=turn                          
seeds=(44 43 42 2 3 4)
cs=(0.1)
ms=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
nms=(0.1)
ss=(30)
wus=(10)

for c in "${cs[@]}"; do
  for m  in "${ms[@]}";  do
    for nm in "${nms[@]}"; do
      for s  in "${ss[@]}"; do
        for wu in "${wus[@]}"; do
          for seed in "${seeds[@]}"; do
            logfile="logs/${DATA}/mm_alpha0_1/gcl/async/hyper/gcl_c${c}_m${m}_nm${nm}_s${s}_wu${wu}_seed${seed}.out"
            [ -f "$logfile" ] && { echo "Skip $logfile"; continue; }

            # throttle concurrent jobs
            while [ "$(pgrep -fc 'python3 ablation.py')" -ge "$MAX_PROCS" ]; do
              sleep 10
            done

            echo "Launching c=$c  m=$m  nm=$nm  s=$s  wu=$wu  seed=$seed"
            nohup python3 ablation.py \
              --dataset "$DATA" \
              --modality multimodal \
              --wm gcl \
              --use_norm_and_cos \
              --alpha "$c" \
              --gcl_m "$m" \
              --noise_mul "$nm" \
              --gcl_s "$s" \
              --drw_warmup "$wu" \
              --seed "$seed" \
              > "$logfile" 2>&1 &

            sleep 0.5
          done
        done
      done
    done
  done
done
"""

"""
for seed in 42 43 44 2 3 4; do
  nohup python3 ablation.py \
    --dataset walk \
    --modality skeleton \
    --wm class_wt \
    --alpha 0 \
    --gcl_m 0 \
    --seed "$seed" \
    > "logs/walk/single/sk_wt_seed${seed}.out" 2>&1 &
  sleep 0.2
done

for seed in 4; do
  nohup python3 ablation.py \
    --synchronized_loading \
    --dataset walk \
    --wm ce \
    --alpha 0 \
    --seed "$seed" \
    > "log_sync.out" 2>&1 &
  sleep 0.2
done
"""
