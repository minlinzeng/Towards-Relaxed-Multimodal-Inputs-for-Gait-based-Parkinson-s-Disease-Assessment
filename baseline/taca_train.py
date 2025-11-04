# file: taca_train.py
# Minimal trainer for a TACA (Temperature-Adjusted Cross-modal Attention) baseline.
import argparse, numpy as np, torch, torch.nn.functional as F
from data.public_pd_datareader import PDReader
from processPdfeData import pdfeReader
from data.efficient_dataloader import create_fusion_loaders
from utility import set_seed, flatten_skel, generate_class_stratified_folds, \
                    get_branch_class_counts, class_weight_tensor, print_class_balance, print_eval_matrix, count_params
from models.taca import TACAWrapper   # single-class wrapper from taca.py

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
    "walk": dict(pose_length=101, sensor_length=65, num_classes=3, lr=1e-3, epochs=100, batch=256),
    "turn": dict(pose_length=101, sensor_length=426, num_classes=3, lr=1e-3, epochs=100, batch=256),
}

# ---------- Training utilities ----------

def forward_step(batch, model, args, sk_w=None, se_w=None):
    x_skel = flatten_skel(batch.get("skeleton"))
    x_sens = batch.get("sensor")
    y_skel = batch.get("label_skeleton")
    y_sens = batch.get("label_sensor")

    if x_skel is not None:
        x_skel = x_skel.to(device).float()
        if x_skel.dim() > 2:
            x_skel = x_skel.view(x_skel.size(0), -1)
    if x_sens is not None:
        x_sens = x_sens.to(device).float()
        if x_sens.dim() > 2:
            x_sens = x_sens.view(x_sens.size(0), -1)
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


def build_model_from_first_batch(train_loader, num_classes, args, hp):
    b0 = next(iter(train_loader))
    x_skel = flatten_skel(b0.get("skeleton"))
    x_sens = b0.get("sensor")

    # infer per-frame dims using known frame counts
    Ts = int(hp["pose_length"])
    Te = int(hp["sensor_length"])
    Ds = int(x_skel[0].numel()) // Ts if x_skel is not None else 1
    De = int(x_sens[0].numel()) // Te if x_sens is not None else 1

    assert x_skel is None or Ds * Ts == int(x_skel[0].numel()), "skeleton shape mismatch"
    assert x_sens is None or De * Te == int(x_sens[0].numel()), "sensor shape mismatch"

    model = TACAWrapper(
        skel_T_frames=Ts, skel_D_frame=Ds,
        sens_T_frames=Te, sens_D_frame=De,
        num_classes=num_classes,
        d_model=args.d_model, n_heads=args.n_heads, n_tok_s=args.n_tok_s, n_tok_e=args.n_tok_e,
        tau=args.tau, gamma=args.gamma, schedule=args.taca_schedule,
        depth_id=0, num_depths=args.taca_depths, dropout=0.1, use_time_shared=True
    ).to(device)

    total = count_params(model)
    print(f"Total params: {total:,} | skel_frame_dim={Ds}, sens_frame_dim={De}, T_s={Ts}, T_e={Te}")
    return model


def train_fold(fold, reader, args, train_subj, eval_subj):
    hp = HP[args.dataset]
    train_loader, eval_loader = create_fusion_loaders(
        args.dataset, reader, train_subj, eval_subj,
        batch_size=hp["batch"], synchronized=args.synced, seed=args.seed,
        num_workers=4, pad_skel=hp["pose_length"], pad_sens=hp["sensor_length"], modality="multimodal"
    )
    print_class_balance(train_loader, hp["num_classes"], tag="TRAIN")
    print_class_balance(eval_loader,  hp["num_classes"], tag="EVAL")

    model = build_model_from_first_batch(train_loader, hp["num_classes"], args, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)

    sk_counts, se_counts = get_branch_class_counts(train_loader, hp["num_classes"])
    sk_w = None if args.wm == "ce" else class_weight_tensor(sk_counts, device)
    se_w = None if args.wm == "ce" else class_weight_tensor(se_counts, device)

    best, best_avg, patience, noimp = {}, -1.0, 60, 0
    for ep in range(1, hp["epochs"]+1):
        model.set_epoch_frac(ep / float(hp["epochs"]))  # for schedule='epoch'

        tl, tsk, tse, *_ = run_epoch(train_loader, model, opt, args, sk_w, se_w, train=True, collect=False)
        vl, vsk, vse, T_sk, P_sk, T_se, P_se, T_ens, P_ens = run_epoch(eval_loader, model, opt, args, sk_w, se_w, train=False, collect=True)

        if args.synced:
            avg = vsk
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | acc {avg:.1f}%")
        else:
            avg = (vsk + vse) / 2.0
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | sk {vsk:.1f}% | se {vse:.1f}% | avg {avg:.1f}%")

        if avg > best_avg:
            best_avg, noimp = avg, 0
            best = dict(T_sk=T_sk, P_sk=P_sk, T_se=T_se, P_se=P_se, T_ens=T_ens, P_ens=P_ens)
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[Fold {fold}] early stop at ep {ep}")
                break

    sk = (np.array(best.get("P_sk", [])) == np.array(best.get("T_sk", []))).mean() * 100.0 if len(best.get("T_sk", [])) else 0.0
    se = (np.array(best.get("P_se", [])) == np.array(best.get("T_se", []))).mean() * 100.0 if len(best.get("T_se", [])) else 0.0

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
    out = np.array(out)
    msk, mse, mav = out.mean(axis=0)
    print(f"\nMean Acc: {mav:.2f}%") if args.synced else \
        print(f"\nMean skel={msk:.2f}%  sensor={mse:.2f}%  avg={mav:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser("TACA baseline trainer")
    p.add_argument("--dataset", choices=["turn","walk"], default="turn")
    p.add_argument("--synced", action="store_true")
    p.add_argument("--wm", choices=["ce","class_wt"], default="ce")
    p.add_argument("--seed", type=int, default=43)

    # TACA knobs
    p.add_argument("--d_model", type=int, default=96)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_tok_s", type=int, default=4, help="#tokens for skeleton")
    p.add_argument("--n_tok_e", type=int, default=4, help="#tokens for sensor")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.5)
    p.add_argument("--taca_schedule", choices=["const","depth","epoch"], default="const")
    p.add_argument("--taca_depths", type=int, default=1)

    main(p.parse_args())

"""
for seed in 42 43 44 2 3 4; do
  nohup python3 baseline/taca_train.py \
    --dataset "turn" \
    --wm "class_wt" \
    --seed "$seed" \
    > "logs/extended_baseline/async/turn/wt_taca_seed${seed}.out" 2>&1 &
  sleep 0.2
done
"""
"""
for seed in 4; do
  nohup python3 baseline/taca_train.py \
    --dataset "walk" \
    --wm "class_wt" \
    --seed "$seed" \
    > "log_walk.out" 2>&1 &
  sleep 0.2
done
"""
