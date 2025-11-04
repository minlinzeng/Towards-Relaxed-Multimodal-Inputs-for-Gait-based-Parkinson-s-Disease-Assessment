import os
import torch
import math
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from data.public_pd_datareader import PDReader
from feature_encoder import MultiModalMultiTaskModel, SkelModalityModel, SensorModalityModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optimizers.weight_methods import CAGrad, PCGrad, FairGrad
from optimizers.losses import LDAMLoss, FocalLoss, GCLLoss
from processPdfeData import pdfeReader
from sklearn.metrics import classification_report
from data.efficient_dataloader import create_fusion_loaders
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')       # use non-interactive backend on servers
import matplotlib.pyplot as plt

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
        "shared_out_channels": 8,
        "backbone_dim": 4,
        "taskhead_input_dim": 8*4,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 64,
        "class_counts": [18, 14, 11]
    },
    "turn": {
        "pose_length": 101,
        "skeleton_input_dim": 21,
        "skeleton_output_dim": 6,
        "sensor_in_channels": 6,
        "sensor_out_channels": 6,
        "sensor_length": 426,
        "shared_out_channels": 10,
        "backbone_dim": 4,
        "taskhead_input_dim": 10*4,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 150,
        "batch_size": 32,
        "class_counts": [6, 25, 3]
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
def choose_model(args, params, use_norm: bool = False, use_cosine: bool = False):
    if args.modality == 'skeleton':
        model = SkelModalityModel(
            skeleton_input_dim   = params["skeleton_input_dim"],
            skeleton_output_dim  = params["skeleton_output_dim"],
            sensor_out_channels  = params["skeleton_output_dim"],  # <— match encoder out
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
            use_norm             = use_norm,
            use_cosine           = use_cosine,
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
def process_batch(batch, model, optimizer, args, sk_counts, se_counts, ldam_skel, ldam_sens, gcl_skel, gcl_sens, grad_method, device, train: bool):
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
    if args.wm == "ldam":
        # LDAMLoss is always non‐negative
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
            l_skel += 0.5*lam*consistency
            l_sens += 0.5*lam*consistency

        if args.modality == "multimodal":
            loss = (l_skel + l_sens) / 2
        elif args.modality == "skeleton":
            loss = l_skel
        else:
            loss = l_sens

        # no extra info dict needed
        info_skel = info_sens = {}
    else:# CE, class‐wt 
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
        if args.modality == "multimodal":
            grad_method.backward(losses=[l_skel, l_sens], shared_parameters=list(model.backbone.parameters()))
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

def run_epoch(loader, model, optimizer, args, sk_counts, se_counts, ldam_skel, ldam_sens, gcl_skel, gcl_sens, grad_method, device, train: bool, collect_preds: bool = False):
    """Loop over one epoch, return stats and optionally predictions for confusion matrices."""
    total_loss = total_sk = total_se = total_n = 0
    trues_skel, preds_skel = [], []
    trues_sens, preds_sens = [], []

    for idx, batch in enumerate(loader, start=1):
        l, cs, ce, n = process_batch(
            batch, model, optimizer, args, sk_counts, se_counts,
            ldam_skel, ldam_sens,
            gcl_skel, gcl_sens,
            grad_method, device, train
        )
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
        return avg_loss, acc_skel, acc_sens, trues_skel, preds_skel, trues_sens, preds_sens
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
    model     = choose_model(args, params, use_norm=(args.wm == "gcl"), use_cosine=(args.wm == "gcl"))
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], momentum=0.9, weight_decay=1e-4) if args.wm == 'gcl' else optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    grad_method = CAGrad(n_tasks=2, device=device, c=args.alpha, max_norm=1.0)
    # grad_method = FairGrad(2, device, alpha=args.alpha, max_norm=args.max_norm)
    
    # 3) Precompute class counts for weighted loss
    sk_counts, se_counts = get_branch_class_counts(train_loader, params["num_classes"])
    if args.wm == "ldam": 
        w_skel = make_inv_freq_weights(sk_counts)
        w_sens = make_inv_freq_weights(se_counts)
        ldam_skel = LDAMLoss(cls_num_list=sk_counts, max_m=args.ldam_m, weight=w_skel, s=args.ldam_s).to(device) # default 30, 0.5
        ldam_sens = LDAMLoss(cls_num_list=se_counts, max_m=args.ldam_m, weight=w_sens, s=args.ldam_s).to(device)
    else:
        ldam_skel = ldam_sens = None
        
    if args.wm.lower() == "gcl":
        w_skel = make_inv_freq_weights(sk_counts)
        w_sens = make_inv_freq_weights(se_counts)
        gcl_skel = GCLLoss(cls_num_list=sk_counts, m=args.gcl_m, s=args.gcl_s, noise_mul=args.noise_mul, weight=None).to(device)
        gcl_sens = GCLLoss(cls_num_list=se_counts, m=args.gcl_m, s=args.gcl_s, noise_mul=args.noise_mul, weight=None).to(device)
    else:
        gcl_skel = gcl_sens = None
    
    best_avg     = 0.0
    best_trues_sk, best_preds_sk, best_trues_se, best_preds_se  = [], [], [], []
    no_improve    = 0
    # patience      = 100
    train_losses = []
    val_losses   = []
    
    # 4) Actual Training Loop
    for ep in range(params["epochs"]): 
        if args.wm.lower() == "gcl" and ep == args.drw_warmup:
            print(f"[Fold {fold_idx}] DRW: applying class weights at epoch {ep+1}")
            # assign the precomputed weight tensors
            gcl_skel.weight = w_skel
            gcl_sens.weight = w_sens

        print(f"\n--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} TRAIN ---")
        model.train()
        tl, tsk, tse = run_epoch(train_loader, model, optimizer, args, sk_counts, se_counts, ldam_skel, ldam_sens, gcl_skel, gcl_sens, grad_method, device, train=True)

        print(f"--- Fold {fold_idx} | Epoch {ep+1}/{params['epochs']} EVAL  ---")
        model.eval()
        with torch.no_grad():
            vl, vsk, vse, t_sk, p_sk, t_se, p_se = run_epoch(eval_loader, model, optimizer, args, sk_counts, se_counts, ldam_skel, ldam_sens, gcl_skel, gcl_sens, grad_method, device, train=False, collect_preds=True)

        train_losses.append(tl)
        val_losses.append(vl)
        scheduler.step(vl)
        avg = (vsk + vse)/2 if args.modality=="multimodal" else (vsk if args.modality=="skeleton" else vse)
        if avg > best_avg:
            best_avg = avg
            best_trues_sk, best_preds_sk = t_sk, p_sk
            best_trues_se, best_preds_se = t_se, p_se
            no_improve = 0
        else:
            no_improve += 1
            # if no_improve >= patience:
            #     print(f"[Fold {fold_idx}] No improvement for {patience} epochs → early stopping at epoch {ep+1}")
            #     break
        print(f"[Fold {fold_idx}][Ep {ep+1}] "
              f"Train loss={tl:.3f} skel={tsk:.1f}% sen={tse:.1f}% | "
              f"Eval loss={vl:.3f} skel={vsk:.1f}% sen={vse:.1f}% avg={avg:.1f}%")

    epochs = list(range(1, len(train_losses)+1))

    # ——— Save Loss Plot ———
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs,   val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold_idx} Loss Curves")
    plt.legend()
    plt.tight_layout()
    out_dir = f"loss_plots/fold_{fold_idx}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{args.dataset}_{args.modality}_{args.wm}_loss_curve.png"))
    plt.close()

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
    return sk_acc, se_acc, best_avg

def main(args):
    set_random_seed(args.seed)
    
    # reader & fold split
    if args.dataset=="walk":
        reader = PDReader(joints_path  = PATHS["walk"]["pose_path"], sensor_path= PATHS["walk"]["sensor_path"], labels_path = PATHS["walk"]["label_path"])
    else:
        reader = pdfeReader(pose_path   = PATHS["turn"]["pose_path"], sensor_path = PATHS["turn"]["sensor_path"], label_path  = PATHS["turn"]["label_path"], lifted_path = PATHS["turn"]["lifted_path"])
    folds = generate_class_stratified_folds(reader, args.dataset)

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
        print(f"→ mean skel={mean_sk:.2f}%, sensor={mean_se:.2f}%, avg={mean_av:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='turn', help="Dataset to use: walk or turn")
    parser.add_argument("--modality", type=str, default='multimodal', help="skeleton, sensor, both, multimodal, or all")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--wm", type=str, default='gcl', help="Weighting method: ce, class_wt, ldam, or gcl")
    parser.add_argument("--synchronized_loading", action="store_true", help="time-align pose+sensor for both train & eval")
    parser.add_argument("--alpha", type=float, default=0.1, help="FairGrad α parameter or CAGrad c parameter")
    parser.add_argument("--max_norm", type=float, default=1.0, help="clip grad norm")
    parser.add_argument("--ldam_s", type=float, default=30)
    parser.add_argument("--ldam_m", type=float, default=0.5)
    parser.add_argument("--gcl_m", type=float, default=0.5)
    parser.add_argument("--gcl_s", type=float, default=30)
    parser.add_argument("--noise_mul", type=float, default=0)
    parser.add_argument("--drw_warmup", type=int, default=10, help="Number of warmup epochs before applying class re-weighting (DRW)")
    print("Arguments: ", parser.parse_args())
    main(parser.parse_args())

"""
for seed in 42 43 44 2 3 4; do
  nohup python3 single_modality.py \
    --dataset turn \
    --modality multimodal \
    --wm gcl \
    --alpha 0.1 \
    --gcl_s 25 \
    --gcl_m 0.1 \
    --noise_mul 0 \
    --drw_warmup 0 \
    --seed "$seed" \
    > "logs/shallower/turn/async/a0.1_s25_m0.1_nm0_wu0_reg_b256_50ep_${seed}.out" 2>&1 &
  sleep 0.2
done

for seed in 42 43 44 2 3 4; do
  nohup python3 single_modality.py \
    --dataset walk \
    --modality multimodal \
    --wm gcl \
    --alpha 0.1 \
    --gcl_s 25 \
    --gcl_m 0.2 \
    --noise_mul 0 \
    --drw_warmup 0 \
    --seed "$seed" \
    > "logs/deeper/walk/async/a0.1_s25_m0.2_nm0_wu0_reg_b64_100ep_${seed}.out" 2>&1 &
  sleep 0.2
done
"""