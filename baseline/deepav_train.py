# file: deepav_train.py
# Compact DeepAVFusion trainer. Uses utility helpers to cut boilerplate.

import argparse, numpy as np, torch, torch.nn.functional as F
from models.deepav import DeepAVLite
from data.public_pd_datareader import PDReader
from processPdfeData import pdfeReader
from data.efficient_dataloader import create_fusion_loaders
from utility import (
    set_seed, flatten_skel, generate_class_stratified_folds,
    get_branch_class_counts, class_weight_tensor,
    print_class_balance, print_eval_matrix, count_params
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- config (unchanged names) ----
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
    "walk": dict(pose_length=101, sensor_length=65,  num_classes=3, lr=1e-3, epochs=100, batch=256),
    "turn": dict(pose_length=101, sensor_length=426, num_classes=3, lr=1e-3, epochs=100, batch=256),
}
# ----------------------------------

def _reader(task: str, seed: int):
    p = PATHS[task]
    if task == "walk":
        return PDReader(joints_path=p["pose_path"], sensor_path=p["sensor_path"], labels_path=p["label_path"])
    return pdfeReader(pose_path=p["pose_path"], sensor_path=p["sensor_path"],
                      label_path=p["label_path"], lifted_path=p.get("lifted_path"),
                      pose_seg=36, sensor_seg=36, downsample_factor=3)


def _build_model(train_loader, num_classes, need_async_heads: bool, args):
    b0 = next(iter(train_loader))
    d_skel = int(flatten_skel(b0["skeleton"]).shape[-1])
    d_sens = int(b0["sensor"].shape[-1])

    # Align CLS usage with --synced
    use_cls = bool(args.synced)
    pool    = "cls" if args.synced else "mean"

    m = DeepAVLite(
        skel_in_dim=d_skel, sens_in_dim=d_sens, num_classes=num_classes,
        embed_dim=12, depth=1, heads=4, mlp_ratio=0.5,   
        skel_patch=1, sens_patch=1, stride=4, drop=0.0,
        n_agg=1, n_fusion=1,
        use_cls=use_cls, pool=pool,
        share_blocks=True, share_unimodal=True, attn_bottleneck=8
    ).to(device)
    
    total = count_params(m)                 # trainable params
    print(f"Total params: {total:,}")

    if need_async_heads and hasattr(m, "_ensure_async_heads"):
        m._ensure_async_heads(num_classes)
    return m


def _criterion(wm: str, loader, num_classes: int):
    if wm == "ce":
        return F.cross_entropy, None, None
    sk_counts, se_counts = get_branch_class_counts(loader, num_classes)
    w_sk = class_weight_tensor(sk_counts, device)
    w_se = class_weight_tensor(se_counts, device)
    return F.cross_entropy, w_sk, w_se

def run_epoch(loader, model, opt, ce_fn, synced: bool, train: bool, collect: bool, w_sk, w_se):
    model.train() if train else model.eval()
    tot_loss, n = 0.0, 0

    # collectors reused by utility.print_eval_matrix
    rec = dict(T_sk=[], P_sk=[]) if synced else dict(T_sk=[], P_sk=[], T_se=[], P_se=[])

    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for b in loader:
            sk = flatten_skel(b["skeleton"]).to(device).float()
            se = b["sensor"].to(device).float()

            if synced:
                y = b["label_skeleton"].to(device).long()
                logits, _ = model(sk, se, synced=True) 
                loss = ce_fn(logits, y, weight=w_sk)
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if collect and not train:
                    rec["T_sk"] += y.cpu().tolist()
                    rec["P_sk"] += logits.argmax(1).cpu().tolist()
            else:
                y_sk = b["label_skeleton"].to(device).long()
                y_se = b["label_sensor"].to(device).long()
                _, sk_pool, se_pool = model.forward_feats(sk, se)   
                log_sk = model.head_skel(sk_pool); log_se = model.head_sens(se_pool)
                loss = ce_fn(log_sk, y_sk, weight=w_sk) + ce_fn(log_se, y_se, weight=w_se)
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if collect and not train:
                    rec["T_sk"] += y_sk.cpu().tolist(); rec["P_sk"] += log_sk.argmax(1).cpu().tolist()
                    rec["T_se"] += y_se.cpu().tolist(); rec["P_se"] += log_se.argmax(1).cpu().tolist()

            tot_loss += float(loss.item()); n += 1

    if synced:
        acc = (np.array(rec["P_sk"]) == np.array(rec["T_sk"])).mean()*100.0 if rec["T_sk"] else 0.0
        return tot_loss/max(1,n), acc, rec
    sk = (np.array(rec["P_sk"]) == np.array(rec["T_sk"])).mean()*100.0 if rec["T_sk"] else 0.0
    se = (np.array(rec["P_se"]) == np.array(rec["T_se"])).mean()*100.0 if rec["T_se"] else 0.0
    return tot_loss/max(1,n), (sk, se), rec

def train_fold(fold, reader, args, train_subj, eval_subj):
    hp = HP[args.dataset]
    train_loader, eval_loader = create_fusion_loaders(
        args.dataset, reader, train_subj, eval_subj,
        batch_size=hp["batch"], synchronized=args.synced, seed=args.seed,
        num_workers=4, pad_skel=hp["pose_length"], pad_sens=hp["sensor_length"], modality="multimodal"
    )
    print_class_balance(train_loader, hp["num_classes"], tag="TRAIN")
    print_class_balance(eval_loader,  hp["num_classes"], tag="EVAL")

    model = _build_model(train_loader, hp["num_classes"], need_async_heads=not args.synced, args=args)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    ce_fn, w_sk, w_se = _criterion(args.wm, train_loader, hp["num_classes"])

    best_score, best = -1.0, {}
    patience, noimp = 60, 0

    for ep in range(1, hp["epochs"]+1):
        tl, _,  _    = run_epoch(train_loader, model, opt, ce_fn, args.synced, True,  False, w_sk, w_se)
        vl, vacc, re = run_epoch(eval_loader,  model, opt, ce_fn, args.synced, False, True, w_sk, w_se)

        if args.synced:
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | acc {vacc:.1f}%")
            score = vacc
        else:
            vsk, vse = vacc; avg = 0.5*(vsk+vse)
            print(f"[Fold {fold}] Ep{ep}: loss {tl:.3f}/{vl:.3f} | sk {vsk:.1f}% | se {vse:.1f}% | avg {avg:.1f}%")
            score = avg

        if score > best_score: best_score, best, noimp = score, re, 0
        else:
            noimp += 1
            if noimp >= patience:
                print(f"[Fold {fold}] early stop at ep {ep}")
                break

    if args.synced:
        print(f"\n>>> Fold {fold} Best Acc: {best_score:.2f}%")
        print_eval_matrix(best, synced=True)
        return best_score, 0.0, best_score
    sk = (np.array(best.get("P_sk", [])) == np.array(best.get("T_sk", []))).mean()*100.0 if best.get("T_sk") else 0.0
    se = (np.array(best.get("P_se", [])) == np.array(best.get("T_se", []))).mean()*100.0 if best.get("T_se") else 0.0
    avg = 0.5*(sk+se)
    print(f"\n>>> Fold {fold} Best skel={sk:.2f}%  sensor={se:.2f}%  avg={avg:.2f}%")
    print_eval_matrix(best, synced=False)
    return sk, se, avg

def main(args):
    set_seed(args.seed)
    reader = _reader(args.dataset, seed=args.seed)
    folds = generate_class_stratified_folds(reader, args.dataset, exclude_subjects=["SUB10","SUB30","SUB22"])

    out = []
    for i, (tr, ev) in enumerate(folds, 1):
        print(f"\n=== Fold {i}/{len(folds)} ===\nTrain: {tr}\nEval : {ev}")
        out.append(train_fold(i, reader, args, tr, ev))
    out = np.array(out)
    if args.synced:
        _, _, mav = out.mean(axis=0); print(f"\nMean Acc: {mav:.2f}%")
    else:
        msk, mse, mav = out.mean(axis=0); print(f"\nMean skel={msk:.2f}%  sensor={mse:.2f}%  avg={mav:.2f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser("DeepAVLite trainer")
    p.add_argument("--dataset", choices=["turn","walk"], default="turn")
    p.add_argument("--synced", action="store_true")
    p.add_argument("--wm", choices=["ce","class_wt"], default="ce")
    p.add_argument("--seed", type=int, default=43)
    main(p.parse_args())

"""
for seed in 42 43 44 2 3 4; do
  nohup python3 baseline/deepav_train.py \
    --dataset "walk" \
    --wm "ce" \
    --seed "$seed" \
    > "logs/extended_baseline/async/walk/deepav_seed${seed}.out" 2>&1 &
  sleep 0.2
done
"""