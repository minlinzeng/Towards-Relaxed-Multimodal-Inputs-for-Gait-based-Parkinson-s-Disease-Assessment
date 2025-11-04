# file: focal_main_min.py
# Minimal trainer for the FOCAL-style shared/private baseline (no pretrain, no CAGrad/GCL).
import argparse, os, numpy as np, torch, torch.nn.functional as F
from models.focal import FOCALSharedLatentBaseline
from data.public_pd_datareader import PDReader
from processPdfeData import pdfeReader
from data.efficient_dataloader import create_fusion_loaders
from utility import set_seed, flatten_skel, generate_class_stratified_folds, \
                    get_branch_class_counts, class_weight_tensor, print_class_balance, print_eval_matrix, count_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = {
    "walk": dict(pose_path="./PD_3D_motion-capture_data/C3Dfiles_processed_new",
                 sensor_path="./PD_3D_motion-capture_data/GRF_processed",
                 label_path="./PD_3D_motion-capture_data/PDGinfo.xlsx"),
    "turn": dict(pose_path="./PD_3D_motion-capture_data/turn-in-place/predictions/",
                 sensor_path="./PD_3D_motion-capture_data/turn-in-place/IMU/",
                 label_path="./PD_3D_motion-capture_data/turn-in-place/PDFEinfo.xlsx",
                 lifted_path="./PD_3D_motion-capture_data/turn-in-place/lifted/"),
}

HP = {
    "walk": dict(pose_length=101, skeleton_input_dim=51, skeleton_output_dim=3,
                 sensor_in_channels=3, sensor_out_channels=3, sensor_length=65,
                 shared_out_channels=16, backbone_dim=8, num_classes=3,
                 lr=1e-3, epochs=100, batch=256),
    "turn": dict(pose_length=101, skeleton_input_dim=21, skeleton_output_dim=6,
                 sensor_in_channels=6, sensor_out_channels=6, sensor_length=426,
                 shared_out_channels=16, backbone_dim=8, num_classes=3,
                 lr=1e-3, epochs=100, batch=256),
}

def forward_step(batch, model, args, sk_w=None, se_w=None):
    x_skel = flatten_skel(batch.get("skeleton")); x_sens = batch.get("sensor")
    y_skel = batch.get("label_skeleton");        y_sens = batch.get("label_sensor")
    if x_skel is not None: x_skel = x_skel.to(device).float()
    if x_sens is not None: x_sens = x_sens.to(device).float()
    if y_skel is not None: y_skel = y_skel.to(device).long()
    if y_sens is not None: y_sens = y_sens.to(device).long()

    synced = args.synced and (x_skel is not None) and (x_sens is not None)
    p_skel, p_sens = model(x_skel, x_sens, synced=synced)

    loss = 0.0
    if synced:
        loss = F.cross_entropy(p_skel, y_skel, weight=sk_w)
    else:
        if p_skel is not None and y_skel is not None:
            loss += F.cross_entropy(p_skel, y_skel, weight=sk_w)
        if p_sens is not None and y_sens is not None:
            loss += F.cross_entropy(p_sens, y_sens, weight=se_w)

    # accuracies
    ns = int(y_skel.size(0)) if y_skel is not None else 0
    ne = int(y_sens.size(0)) if y_sens is not None else 0
    cs = int((p_skel.argmax(1) == y_skel).sum().item()) if (p_skel is not None and ns) else 0
    ce = int((p_sens.argmax(1) == y_sens).sum().item()) if (p_sens is not None and ne) else 0

    return loss, cs, ce, ns, ne, p_skel, p_sens, synced

def run_epoch(loader, model, opt, args, sk_w, se_w, train=False, collect=False):
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.inference_mode()

    tot_loss = cs = ce = ns = ne = 0
    T_sk, P_sk, T_se, P_se, T_ens, P_ens = [], [], [], [], [], []

    with ctx:
        for batch in loader:
            loss, _cs, _ce, _ns, _ne, log_sk, log_se, synced = forward_step(batch, model, args, sk_w, se_w)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tot_loss += float(loss.item()); cs += _cs; ce += _ce; ns += _ns; ne += _ne

            if collect and not train:
                if log_sk is not None and _ns:
                    T_sk += batch["label_skeleton"].tolist()
                    P_sk += log_sk.argmax(1).cpu().tolist()
                if (not synced) and log_se is not None and _ne:
                    T_se += batch["label_sensor"].tolist()
                    P_se += log_se.argmax(1).cpu().tolist()
                if (not synced) and log_sk is not None and log_se is not None:
                    ens = (F.softmax(log_sk, 1) + F.softmax(log_se, 1)) * 0.5
                    T_ens += batch["label_skeleton"].tolist()
                    P_ens += ens.argmax(1).cpu().tolist()

    return (tot_loss / max(1, len(loader)),
            (cs / max(1, ns)) * 100.0,
            (ce / max(1, ne)) * 100.0 if not args.synced else 0.0,
            T_sk, P_sk, T_se, P_se, T_ens, P_ens)


def train_fold(fold, reader, args, train_subj, eval_subj):
    hp = HP[args.dataset]
    train_loader, eval_loader = create_fusion_loaders(
        args.dataset, reader, train_subj, eval_subj,
        batch_size=hp["batch"], synchronized=args.synced, seed=args.seed,
        num_workers=4, pad_skel=hp["pose_length"], pad_sens=hp["sensor_length"], modality="multimodal"
    )
    sk_tr, se_tr = print_class_balance(train_loader, hp["num_classes"], tag="TRAIN")
    sk_ev, se_ev = print_class_balance(eval_loader,  hp["num_classes"], tag="EVAL")
    model = FOCALSharedLatentBaseline(
        skeleton_input_dim=hp["skeleton_input_dim"], skeleton_output_dim=hp["skeleton_output_dim"],
        sensor_in_channels=hp["sensor_in_channels"], sensor_out_channels=hp["sensor_out_channels"],
        sensor_length=hp["sensor_length"], d_shared=16, d_private=8,
        shared_out_channels=4, backbone_dim=4,
        num_classes=hp["num_classes"], use_norm_head=False, use_cosine_head=False
    ).to(device)
    
    total = count_params(model)                 # trainable params
    print(f"Total params: {total:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)

    sk_counts, se_counts = get_branch_class_counts(train_loader, hp["num_classes"])
    sk_w = None if args.wm == "ce" else class_weight_tensor(sk_counts, device)
    se_w = None if args.wm == "ce" else class_weight_tensor(se_counts, device)

    best, best_avg, patience, noimp = {}, -1.0, 60, 0
    for ep in range(1, hp["epochs"]+1):
        tl, tsk, tse, *_ = run_epoch(train_loader, model, opt, args, sk_w, se_w, train=True, collect=False)
        vl, vsk, vse, T_sk, P_sk, T_se, P_se, T_ens, P_ens = run_epoch(eval_loader, model, opt, args, sk_w, se_w, train=False, collect=True)
        if args.synced:
            avg = vsk
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | acc {avg:.1f}%")
        else:
            avg = (vsk + vse)/2.0
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | sk {vsk:.1f}% | se {vse:.1f}% | avg {avg:.1f}%")
        if avg > best_avg:
            best_avg, noimp = avg, 0
            best = dict(T_sk=T_sk, P_sk=P_sk, T_se=T_se, P_se=P_se, T_ens=T_ens, P_ens=P_ens)
            # os.makedirs(args.ckpt, exist_ok=True)
            # save_checkpoint(model, os.path.join(args.ckpt, f"focal_fold{fold}.pt"))
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[Fold {fold}] early stop at ep {ep}")
                break

    sk = (np.array(best.get("P_sk", [])) == np.array(best.get("T_sk", []))).mean()*100.0 if len(best.get("T_sk", [])) else 0.0
    se = (np.array(best.get("P_se", [])) == np.array(best.get("T_se", []))).mean()*100.0 if len(best.get("T_se", [])) else 0.0

    print(f"\n>>> Fold {fold} Best Mean Acc: {best_avg:.2f}%")
    print_eval_matrix(best, args.synced)

    if args.synced:
        print(f"\n*** Fold {fold} Best Acc: {sk:.2f}% ***\n")
        return sk, 0.0, sk
    else:
        avg = (sk + se) / 2.0
        print(f"\n*** Fold {fold} Best skel={sk:.2f}%  sensor={se:.2f}%  avg={avg:.2f}% ***\n")
        return sk, se, avg
    
def main(args):
    set_seed(args.seed)
    if args.dataset == "walk":
        reader = PDReader(joints_path=PATHS["walk"]["pose_path"],
                          sensor_path=PATHS["walk"]["sensor_path"],
                          labels_path=PATHS["walk"]["label_path"])
    else:
        reader = pdfeReader(pose_path=PATHS["turn"]["pose_path"],
                            sensor_path=PATHS["turn"]["sensor_path"],
                            label_path=PATHS["turn"]["label_path"],
                            lifted_path=PATHS["turn"]["lifted_path"])
    folds = generate_class_stratified_folds(reader, args.dataset, exclude_subjects=["SUB10","SUB30","SUB22"])
    out = []
    for i, (tr, ev) in enumerate(folds, 1):
        print(f"\n=== Fold {i}/{len(folds)} ===\nTrain: {tr}\nEval : {ev}")
        out.append(train_fold(i, reader, args, tr, ev))
    out = np.array(out)  # shape: (num_folds, 3)
    msk, mse, mav = out.mean(axis=0)
    print(f"\nMean Acc: {mav:.2f}%") if args.synced else \
        print(f"\nMean skel={msk:.2f}%  sensor={mse:.2f}%  avg={mav:.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser("FOCAL minimal trainer")
    p.add_argument("--dataset", choices=["turn","walk"], default="turn")
    p.add_argument("--synced", action="store_true", help="Require paired inputs; enables ensemble & KL consistency")
    p.add_argument("--wm", choices=["ce","class_wt"], default="ce")
    p.add_argument("--seed", type=int, default=43)
    # p.add_argument("--use_norm_and_cos", action="store_true")
    p.add_argument("--consistency_lambda", type=float, default=1.0)
    main(p.parse_args())


"""
for seed in 42 43 44 2 3 4; do
  nohup python3 baseline/focal_train.py \
    --wm "ce" \
    --dataset "walk" \
    --seed "$seed" \
    > "logs/extended_baseline/async/walk/focal_seed${seed}.out" 2>&1 &
  sleep 0.2
done
"""
"""
for seed in 4; do
  nohup python3 baseline/focal_train.py \
    --wm "class_wt" \
    --dataset "walk" \
    --seed "$seed" \
    > "log.out" 2>&1 &
  sleep 0.2
done
"""