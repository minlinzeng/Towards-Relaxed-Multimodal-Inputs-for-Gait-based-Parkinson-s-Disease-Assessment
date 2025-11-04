#!/usr/bin/env python3
import argparse, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from collections import Counter
from sklearn.metrics import classification_report
from processPdfeData import pdfeReader
from Dataloader import create_balanced_fusion_loaders      
from Models import MultiModalMultiTaskModel, EarlyFusionModel, LateFusionModel, ShareLatentModel, CheapXAttnModel, SkelOnlyModel, SensorOnlyModel
from collections import defaultdict
import random
import torch.nn.functional as F

# ─── Config ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = dict(
    turn = dict(
        pose_path   = './PD_3D_motion-capture_data/turn-in-place/predictions/',
        lifted_path = './PD_3D_motion-capture_data/turn-in-place/lifted/',
        sensor_path = './PD_3D_motion-capture_data/turn-in-place/IMU/',
        label_path  = './PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx'
    )
)

PARAMS = dict(
    skeleton_input_dim   = 21,
    skeleton_output_dim  = 6,
    sensor_in_channels   = 6,
    sensor_out_channels  = 6,
    sensor_length        = 150,
    pose_length          = 101,
    shared_out_channels  = 16,
    backbone_dim         = 8,
    taskhead_dim         = 16*8,   # shared_out * backbone_dim
    num_classes          = 3,
    learning_rate        = 1e-3,
    epochs               = 50,
    batch_size           = 256,
    num_workers          = 4,
)

MODEL_CLASSES = {
    'multitask':    MultiModalMultiTaskModel,
    'early':        EarlyFusionModel,
    'late':         LateFusionModel,
    'share_latent': ShareLatentModel,
    'cheap_xattn':  CheapXAttnModel,
    'skeleton':     SkelOnlyModel,
    'sensor':       SensorOnlyModel,
}

FUSION_MODELS = {
    "early":        EarlyFusionModel,
    "late":         LateFusionModel,
    "share_latent": ShareLatentModel,
    "cheap_xattn":  CheapXAttnModel,
}

# ─── Utils ────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def flatten_skel(x):
    # x: (B, T, J, C) → (B, T, J*C)
    if x.dim()==4:
        B,T,J,C = x.shape
        return x.view(B, T, J*C)
    return x

def make_stratified_folds(reader):
    """
    For a TURN‐style reader.labels_dict:
      • build subj→label
      • group subjects by class
      • let m = size of smallest class
      • for each class, trim/shuffle its list to length m
      • for i in 0..m-1, hold out one subject per class as the eval set
    Returns List[(train_subjects, eval_subjects)] of length m.
    """
    # 1) subject→label
    subj2lbl = {
        s: (v[0] if isinstance(v,(list,tuple)) else int(v))
        for s,v in reader.labels_dict.items()
        if s not in ("SUB10","SUB30","SUB22")
    }
    # 2) group by class
    cls2subs = defaultdict(list)
    for s,lbl in subj2lbl.items():
        cls2subs[lbl].append(s)
    # 3) find smallest class size
    m = min(len(v) for v in cls2subs.values())
    # 4) downsample & shuffle each class to m subjects
    for lbl, subs in cls2subs.items():
        random.shuffle(subs)
        cls2subs[lbl] = subs[:m]
    # 5) build folds
    folds = []
    for i in range(m):
        eval_subj = [ cls2subs[lbl][i] for lbl in sorted(cls2subs) ]
        train_subj = [s for s in subj2lbl if s not in eval_subj]
        folds.append((train_subj, eval_subj))
    return folds


def build_model(model_name: str, cfg: dict, device):
    """
    Instantiate and return the right model (moved to `device`),
    given:
      - model_name in {"skeleton","sensor", *fusion keys*}
      - cfg: dict of hyperparams (e.g. your P)
    """
    # single-modality
    if model_name == "skeleton":
        return SkelOnlyModel(
            cfg["skeleton_input_dim"],
            cfg["skeleton_output_dim"],
            cfg["shared_out_channels"],
            cfg["backbone_dim"],
            cfg["num_classes"]
        ).to(device)

    if model_name == "sensor":
        return SensorOnlyModel(
            cfg["sensor_in_channels"],
            cfg["sensor_out_channels"],
            cfg["sensor_length"],
            cfg["shared_out_channels"],
            cfg["backbone_dim"],
            cfg["num_classes"]
        ).to(device)

    # multi-modal fusion
    FusionClass = FUSION_MODELS[model_name]
    return FusionClass(
        skeleton_input_dim   = cfg["skeleton_input_dim"],
        skeleton_output_dim  = cfg["skeleton_output_dim"],
        sensor_in_channels   = cfg["sensor_in_channels"],
        sensor_out_channels  = cfg["sensor_out_channels"],
        sensor_length        = cfg["sensor_length"],
        shared_out           = cfg["shared_out_channels"],
        backbone_dim         = cfg["backbone_dim"],
        num_classes          = cfg["num_classes"],
        synchronized_loading = True
    ).to(device)



def run_epoch(loader, model, criterion, optimizer=None, collect=False, eval_mode='full'):
    """
    eval_mode ∈ {'full','skel','sens'}:
      - 'full': average softmax(p_skel)+softmax(p_sens)
      - 'skel': only p_skel
      - 'sens': only p_sens
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = correct = total = 0
    trues, preds = [], []

    for batch in loader:
        sk = flatten_skel(batch["skeleton"].to(DEVICE)).float()
        se = batch["sensor"].to(DEVICE).float()
        y  = batch["label_skeleton"].to(DEVICE)  # same as label_sensor under sync

        if not is_train:
            model.use_skel_only = (eval_mode == 'skel')
            model.use_sens_only = (eval_mode == 'sens')

        with torch.set_grad_enabled(is_train):
            out = model(sk, se)
            p_skel, p_sens = out if isinstance(out, tuple) else (out, None)

            # pick logits per eval_mode
            if eval_mode=='full':
                ps = F.softmax(p_skel, dim=1)
                pt = F.softmax(p_sens, dim=1) if p_sens is not None else ps
                logits = (ps + pt) / 2
            elif eval_mode=='skel':
                logits = p_skel
            elif eval_mode=='sens':
                logits = p_sens if p_sens is not None else p_skel

            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if not is_train:
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
            if collect:
                trues .extend(y.cpu().tolist())
                preds.extend(pred.cpu().tolist())

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    acc      = correct / total * 100 if (not is_train) else None
    if collect and not is_train:
        return avg_loss, acc, trues, preds
    return avg_loss, acc


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("SYNC-only TURN: single vs fusion")
    parser.add_argument("model", choices=list(MODEL_CLASSES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    reader = pdfeReader(**PATHS["turn"])
    folds  = make_stratified_folds(reader)
    all_acc = []

    for fold_idx, (tr_subj, ev_subj) in enumerate(folds, 1):
        print(f"\n=== Fold {fold_idx}/{len(folds)} ({args.model}) ===")
        tl, el = create_balanced_fusion_loaders(
            reader, tr_subj, ev_subj,
            batch_size=PARAMS["batch_size"],
            pad_skel=PARAMS["pose_length"],
            pad_sens=PARAMS["sensor_length"],
            seed=args.seed
        )

        model     = build_model(args.model, PARAMS, DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

        best_acc   = 0.0
        best_state = None

        # — training + inline full‐val for checkpointing —
        for ep in range(1, PARAMS["epochs"]+1):
            t_loss, _    = run_epoch(tl, model, criterion, optimizer)
            _, e_full, _, _ = run_epoch(el, model, criterion, None, collect=True, eval_mode='full')
            if e_full > best_acc:
                best_acc   = e_full
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            print(f"[Fold {fold_idx}][Ep {ep}/{PARAMS['epochs']}] " 
                  f"Val(full)={e_full:5.2f}%")

        # reload best
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        # — final evals —
        _, full_acc, tr_full, pr_full = run_epoch(el, model, criterion, None, collect=True, eval_mode='full')
        _, sk_acc,   tr_skel, pr_skel = run_epoch(el, model, criterion, None, collect=True, eval_mode='skel')
        _, se_acc,   tr_sens, pr_sens = run_epoch(el, model, criterion, None, collect=True, eval_mode='sens')

        print(f"\n--- Fold {fold_idx} final EVAL ---")
        print(f" Full MM       : {full_acc:6.2f}%")
        print(f" Skeleton-only : {sk_acc:6.2f}%")
        print(f" Sensor-only   : {se_acc:6.2f}%\n")

        print(" Full MM Report:")
        print(classification_report(tr_full, pr_full, digits=2, zero_division=0))

        all_acc.append(full_acc)

    print(f"\n>>> Mean over {len(folds)} folds: {np.mean(all_acc):5.2f}% <<<")
    

if __name__ == "__main__":
    main()
    
"""
MODELS=(early late share_latent cheap_xattn)
SEEDS=(42 43 44 2 3 4)
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    nohup python3 Modality_Collapse/fusion_train.py $MODEL --seed $SEED > logs/collapse/mask_infer/${MODEL}_${SEED}.log 2>&1 &
    sleep 0.1
  done
done
"""
