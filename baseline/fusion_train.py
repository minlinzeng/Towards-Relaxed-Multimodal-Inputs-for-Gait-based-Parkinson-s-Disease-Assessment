import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from collections import Counter
from data.public_pd_datareader import PDReader
from processPdfeData import pdfeReader
from feature_encoder import EarlyFusionModel, LateFusionModel, ShareLatentModel, CheapXAttnModel
from data.efficient_dataloader import create_fusion_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = {
    "walk": {
        "pose_path": "./PD_3D_motion-capture_data/C3Dfiles_processed_new",  
        # "pose_path": "./PD_3D_motion-capture_data/C3Dfiles_cleaned_sequences",
        "sensor_path": "./PD_3D_motion-capture_data/GRF_processed",
        "label_path": "./PD_3D_motion-capture_data/PDGinfo.xlsx",
    },
    "turn": {
        "pose_path": './PD_3D_motion-capture_data/turn-in-place/predictions/',
        "lifted_path": './PD_3D_motion-capture_data/turn-in-place/lifted/',
        "sensor_path": './PD_3D_motion-capture_data/turn-in-place/IMU/',
        "label_path": './PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx'
    }
}

MODALITY_PARAMS = {
    "walk": {
        "skeleton_input_dim": 51,
        "skeleton_output_dim": 3,
        "sensor_in_channels": 3,
        "sensor_out_channels": 3,
        "sensor_length": 65,
        "pose_length": 101,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 32,  
    },
    "turn": {
        "skeleton_input_dim": 21,
        "skeleton_output_dim": 6,
        "sensor_in_channels": 6,
        "sensor_out_channels": 6,
        "sensor_length": 150,
        "pose_length": 101,
        "shared_out_channels": 16,
        "backbone_dim": 8,
        "num_classes": 3,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 256,
    }
}

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
    """
    - walk: only subjects with both pose & sensor go into eval folds;
            train = the rest of those (still filtered later for missing data)
    - turn: unchanged behavior
    """
    from collections import defaultdict
    import random

    if dataset == "walk":
        # 1) find prefixes that truly have both modalities
        pose_pfx = { "_".join(k.split("_")[:2]) for k in reader.pose_dict.keys() }
        sens_pfx = { "_".join(k.split("_")[:2]) for k in reader.sensor_dict.keys() }
        both_modal = pose_pfx & sens_pfx

        # 2) only keep labels for those both‐modal subjects
        raw_labels = reader.pose_label_dict
        label_dict = { s: raw_labels[s] for s in raw_labels if s in both_modal }

    elif dataset == "turn":
        label_dict = {
            s: reader.labels_dict[s][0]
            for s in reader.labels_dict
            if s not in ("SUB10","SUB30","SUB22")
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 3) group subjects by class
    class_to_subj = defaultdict(list)
    for subj, lbl in label_dict.items():
        class_to_subj[lbl].append(subj)

    # 4) number of folds = size of smallest class
    m = min(len(lst) for lst in class_to_subj.values())
    if m == 0:
        raise ValueError("Need ≥1 subject per class with both modalities for walk")

    # 5) balance up/down to m subjects per class
    balanced = {}
    for lbl, subs in class_to_subj.items():
        if len(subs) > m:
            subs = random.sample(subs, k=m)
        random.shuffle(subs)
        balanced[lbl] = subs

    # 6) build the folds
    folds = []
    for i in range(m):
        eval_subj = [balanced[lbl][i] for lbl in sorted(balanced)]
        train_subj = [s for s in label_dict if s not in eval_subj]
        folds.append((train_subj, eval_subj))

    return folds


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

    # choose loss(es)
    if args.synchronized_loading and args.fusion_type != "share_latent":
        criterion = nn.CrossEntropyLoss()
    else:
        # for share_latent-sync or any async mode, we need two heads
        criterion_skel = nn.CrossEntropyLoss()
        criterion_sens = nn.CrossEntropyLoss()

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
        synchronized_loading = args.synchronized_loading
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

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    best_avg_acc = 0.0
    best_skel_acc = 0.0
    best_sens_acc = 0.0

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

    print("-" * 60 + "\n")
    return best_avg_acc, best_skel_acc, best_sens_acc


def main(args):
    # 1) Reproducibility
    set_random_seed(args.seed)

    # 2) Instantiate the appropriate reader
    reader_cls, reader_args = {
        "walk": (PDReader, (
            PATHS["walk"]["pose_path"],
            PATHS["walk"]["sensor_path"],
            PATHS["walk"]["label_path"],
        )),
        "turn": (pdfeReader, (
            PATHS["turn"]["pose_path"],
            PATHS["turn"]["sensor_path"],
            PATHS["turn"]["label_path"],
            PATHS["turn"]["lifted_path"],
        )),
    }[args.dataset]
    reader = reader_cls(*reader_args)

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

    # 8) Show which subjects were held out in each fold
    print("=" * 40)
    print(" EVAL SUBJECTS PER FOLD ")
    print("=" * 40)
    for i, (_, eval_subj) in enumerate(folds, start=1):
        print(f" Fold {i:>2}: {eval_subj}")
    print()

if __name__ == "__main__":
    """
        experiemnts concerning using four fusion methods under
            1. asynchronous loading
            2. synchronized loading
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="walk", choices=["walk","turn"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fusion_type", choices=["early","late","share_latent","cheap_xattn"], default="cheap_xattn", help="early: concat→backbone→two-heads  late: two branches→fuse logits")
    parser.add_argument("--synchronized_loading", action="store_true", help="use time-synchronized sampling for both train and eval")
    args = parser.parse_args()
    main(args)

# nohup python baseline/fusion_train.py --fusion_type early > log_async_no_wt_early.out 2>&1 &
"""
seeds=(42 43 44 2 3 4)
for seed in "${seeds[@]}"; do
  echo "Starting fusion_train.py with fusion_type=late seed=${seed}"
  nohup python3 baseline/fusion_train.py \
    --dataset turn \
    --fusion_type late \
    # --synchronized_loading \
    --seed "${seed}" \
  > "baseline/logs/async_train_async_test/log_lf_s${seed}.out" 2>&1 &
done

pgrep -af fusion_train.py
pkill -f fusion_train.py
"""


"""
for fusion in share_latent; do
  for seed in 42 43 44 2 3 4; do
    nohup python3 baseline/fusion_train.py \
      --dataset turn \
      --synchronized_loading \
      --fusion_type "$fusion" \
      --seed "$seed" \
    > "logs/turn/mm_alpha0_1/fusion/sync_no_wl/0607_${fusion}_256_s${seed}.out" 2>&1 &
  done
done

for fusion in early late share_latent cheap_xattn; do
  for seed in 42 43 44 2 3 4; do
    nohup python3 baseline/fusion_train.py \
      --dataset walk \
      --fusion_type "$fusion" \
      --seed "$seed" \
    > "logs/walk/both_imbalance/fuse/async/${fusion}_32_s${seed}.out" 2>&1 &
  done
done
"""