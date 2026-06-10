# weargait_train.py
"""
Three‐stream training for WearGait with CAGrad (shared) + GCL/CE per modality.
- Walkway, Insole, IMU are separate branches.
- Shared backbone; CAGrad mixes grads on shared params only.
- Private params updated by their own loss.
- Sync: one common head + one label. Async: three heads + three labels.

Requires:
  - data_processing/dataloader_weargait.py: prepare_split, make_sync_loaders, make_async_loaders,
                         make_fixed_balanced_folds_no_overlap, build_subj2label
  - feature_encoder_weargait.py: WearGaitThreeModal with:
        get_shared_parameters()
        walkway_parameters(), insole_parameters(), imu_parameters()
  - optimizers.classification_losses.GCLLoss, optimizers.multitask_weighting.CAGrad
  - learning.training_common.set_seed
"""

import argparse, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEARGAIT_ROOT = PROJECT_ROOT / "data" / "WearGait"
for path in (PROJECT_ROOT, Path(__file__).resolve().parent, WEARGAIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from learning.training_common import set_seed
from data_processing.dataloader_weargait import (
    prepare_split, make_sync_loaders, make_async_loaders,
    make_fixed_balanced_folds_no_overlap, build_subj2label,
)
from weargait_encoders import WearGaitThreeModal, EarlyFusion3, LateFusion3, SharedLatent3, CheapXAttn3
from learning.optimizers.multitask_weighting import CAGrad
from learning.optimizers.classification_losses import GCLLoss
from baselines.architectures.deepav import DeepAVLite3
from baselines.architectures.focal import FOCALSharedLatent3
from baselines.architectures.taca import TACA3TriWrapper
import torch.nn.functional as F
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hc_path = "data/WearGait/HC"
pd_path = "data/WearGait/PD"
output_dir = "data/WearGait/WearGait_preproc_SPmT_30Hz"

MASK_COMBOS = {
    "W"     : (True,  False, False),
    "I"     : (False, True,  False),
    "M"     : (False, False, True ),
    "W+I"   : (True,  True,  False),
    "W+M"   : (True,  False, True ),
    "I+M"   : (False, True,  True ),
    "W+I+M" : (True,  True,  True ),
}

# ---------- subject discovery ----------
def _scan_subjects(dir_path: str) -> list[str]:
    p = Path(dir_path)
    return sorted({x.name.split("_")[0] for x in p.glob("*_matTURN.csv")})

def discover_pd_hc(pd_dir: str, hc_dir: str) -> tuple[list[str], list[str]]:
    pd_ids = _scan_subjects(pd_dir)
    hc_ids = _scan_subjects(hc_dir)
    if not pd_ids or not hc_ids:
        raise ValueError("No subjects found under pd_dir/hc_dir.")
    return pd_ids, hc_ids

# ---- quick sample counters (paste in train.py) ----
# def report_counts(prep, tr_loader, te_loader, async_mode, modalities=("walkway","insole","imu")):
    if async_mode:
        ds_tr = tr_loader.dataset
        ds_te = te_loader.dataset
        # epoch size = min per-modality len (no-dup within epoch)
        tr_lens = {m: ds_tr._lens_full[m] for m in modalities}
        te_lens = {m: ds_te._lens_full[m] for m in modalities}
        print(f"[ASYNC] train per-modality windows: {tr_lens}  → epoch_samples={len(ds_tr)}")
        print(f"[ASYNC]  test per-modality windows: {te_lens}  → epoch_samples={len(ds_te)}")
    else:
        # sync uses precomputed aligned pairs
        print(f"[SYNC] train aligned pairs: {len(prep['train_sync'])}")
        print(f"[SYNC]  test aligned pairs: {len(prep['test_sync'])}")
        # (optional) show raw counts before alignment
        tr_raw = {m: len(prep['train_stores'][m]) for m in modalities}
        te_raw = {m: len(prep['test_stores'][m])  for m in modalities}
        print(f"[SYNC] raw train per-modality windows: {tr_raw}")
        print(f"[SYNC] raw  test per-modality windows: {te_raw}")

# ---------- counting and weighting ----------
@torch.no_grad()
def class_counts_per_mod(loader, num_classes: int, async_mode: bool):
    from collections import Counter
    Cw = Counter(); Ci = Counter(); Cm = Counter()
    for b in loader:
        if async_mode:
            Cw.update(b["y"]["walkway"].cpu().tolist())
            Ci.update(b["y"]["insole" ].cpu().tolist())
            Cm.update(b["y"]["imu"    ].cpu().tolist())
        else:
            y = b["y"].cpu().tolist()
            Cw.update(y); Ci.update(y); Cm.update(y)
    def to_list(C): return [C[i] for i in range(num_classes)]
    return {"walkway": to_list(Cw), "insole": to_list(Ci), "imu": to_list(Cm)}

def inv_freq_weights(counts: list[int]) -> torch.Tensor:
    w = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=DEVICE) + 1e-8)
    return w / w.sum() * len(counts)

def make_criteria(args, counts):
    wm = args.wm.lower()
    if wm == "gcl":
        Lw = GCLLoss(cls_num_list=counts["walkway"], m=args.gcl_m, s=args.gcl_s,
                     noise_mul=args.noise_mul, weight=None).to(DEVICE)
        Li = GCLLoss(cls_num_list=counts["insole" ], m=args.gcl_m, s=args.gcl_s,
                     noise_mul=args.noise_mul, weight=None).to(DEVICE)
        Lm = GCLLoss(cls_num_list=counts["imu"    ], m=args.gcl_m, s=args.gcl_s,
                     noise_mul=args.noise_mul, weight=None).to(DEVICE)
        return Lw, Li, Lm
    if wm == "class_wt":
        ww = inv_freq_weights(counts["walkway"])
        wi = inv_freq_weights(counts["insole"])
        wm_ = inv_freq_weights(counts["imu"])
        return (nn.CrossEntropyLoss(weight=ww),
                nn.CrossEntropyLoss(weight=wi),
                nn.CrossEntropyLoss(weight=wm_))
    # plain CE
    ce = nn.CrossEntropyLoss()
    return ce, ce, ce

def build_criteria_and_cagrad(args, tr_loader, async_mode: bool):
    counts = class_counts_per_mod(tr_loader, args.num_classes, async_mode)

    # single-modality baselines
    if args.single_mod is not None:
        mod = args.single_mod  # "walkway" | "insole" | "imu"
        wm = args.wm.lower()
        if wm == "class_wt":
            w = inv_freq_weights(counts[mod])
            return (nn.CrossEntropyLoss(weight=w),), None
        if wm == "gcl":
            L = GCLLoss(cls_num_list=counts[mod], m=args.gcl_m, s=args.gcl_s,
                        noise_mul=args.noise_mul, weight=None).to(DEVICE)
            return (L,), None
        return (nn.CrossEntropyLoss(),), None

    # multi-modal models (conventional + advanced + ours)
    criterions = make_criteria(args, counts)     # supports "class_wt", "gcl", "ce"
    cagrad = None if (args.baseline is not None) else \
             (CAGrad(n_tasks=3, device=DEVICE, c=args.alpha) if args.alpha > 0 else None)
    return criterions, cagrad

def apply_drw_if_needed(ep, args, criterions, counts):
    if args.wm.lower() != "gcl": return
    if ep == (args.drw_warmup + 1):  # apply AFTER warmup epochs
        Lw, Li, Lm = criterions
        Lw.weight = inv_freq_weights(counts["walkway"])
        Li.weight = inv_freq_weights(counts["insole"])
        Lm.weight = inv_freq_weights(counts["imu"])

# ---------- forward helpers ----------
def forward_batch(model, batch, async_mode: bool):
    if async_mode:
        xw = batch["walkway"].to(DEVICE).float()
        xi = batch["insole" ].to(DEVICE).float()
        xm = batch["imu"    ].to(DEVICE).float()
        yw = batch["y"]["walkway"].to(DEVICE).long()
        yi = batch["y"]["insole" ].to(DEVICE).long()
        ym = batch["y"]["imu"    ].to(DEVICE).long()
    else:
        xw, xi, xm = [t.to(DEVICE).float() for t in batch["xs"]]
        y = batch["y"].to(DEVICE).long()
        yw = yi = ym = y
        
    # ---- TACA3Tri path (flatten + pass synced flag) ----
    if getattr(model, "_is_taca3tri", False):
        def _flat(t):
            return None if t is None else t.reshape(t.size(0), -1)
        lw, li, lm = model(_flat(xw), _flat(xi), _flat(xm), synced=(not async_mode))  # sync loader → synced=True
        return (lw, li, lm), (yw, yi, ym)
    
    lw, li, lm = model(xw, xi, xm)  # three logits
    return (lw, li, lm), (yw, yi, ym)

# ---------- optimization step ----------
def step_cagrad_three(model, Lw, Li, Lm, optimizer, cagrad):
    """
    One step of optimization with optional CAGrad:
      • If cagrad is provided and the model exposes accessors:
          - CAGrad mixes gradients on model.get_shared_parameters()
          - Each private stream gets grads from its own loss only
      • Otherwise (baselines / single-mod / no CAGrad):
          - Do a plain averaged backward over available losses

    Lw, Li, Lm can be None when a stream is unused.
    """
    optimizer.zero_grad(set_to_none=True)

    # Collect available losses (ignore None or non-finite just in case)
    losses = [L for L in (Lw, Li, Lm) if (L is not None and torch.isfinite(L))]
    if not losses:
        return  # nothing to do

    use_cagrad = (
        cagrad is not None and
        hasattr(model, "get_shared_parameters")
    )

    if use_cagrad:
        # ---- 1) CAGrad on SHARED params only ----
        shared = list(model.get_shared_parameters())
        if shared:  # only call if there actually are shared params
            cagrad.backward(losses=losses, shared_parameters=shared)

        # ---- 2) PRIVATE params: per-stream grads from its own loss ----
        # Walkway
        if Lw is not None and hasattr(model, "walkway_parameters"):
            priv_w = list(model.walkway_parameters())
            if priv_w:
                gw = torch.autograd.grad(Lw, priv_w, retain_graph=True, allow_unused=True)
                for p, g in zip(priv_w, gw):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad.add_(g)

        # Insole
        if Li is not None and hasattr(model, "insole_parameters"):
            priv_i = list(model.insole_parameters())
            if priv_i:
                gi = torch.autograd.grad(Li, priv_i, retain_graph=True, allow_unused=True)
                for p, g in zip(priv_i, gi):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad.add_(g)

        # IMU
        if Lm is not None and hasattr(model, "imu_parameters"):
            priv_m = list(model.imu_parameters())
            if priv_m:
                gm = torch.autograd.grad(Lm, priv_m, retain_graph=False, allow_unused=True)
                for p, g in zip(priv_m, gm):
                    if g is not None:
                        p.grad = g if p.grad is None else p.grad.add_(g)

    else:
        # ---- No CAGrad (baselines / single-mod) → plain averaged CE ----
        torch.stack(losses).mean().backward()

    optimizer.step()

# ---------- single-modality training (for TRIP comparison) ----------

def _single_logits_and_labels(model, batch, async_mode: bool, mod: str):
    if async_mode:
        x = batch[mod].to(DEVICE).float()
        y = batch["y"][mod].to(DEVICE).long()
    else:
        # sync loader packs as batch["xs"] = [walkway, insole, imu], batch["y"]
        idx = {"walkway":0, "insole":1, "imu":2}[mod]
        x   = batch["xs"][idx].to(DEVICE).float()
        y   = batch["y"].to(DEVICE).long()
    # run only that branch: enc_? -> shared backbone -> head_?
    if mod == "walkway":
        feat = model.backbone(model.enc_w(x)).flatten(1)
        logits = model.head_w(feat)
    elif mod == "insole":
        feat = model.backbone(model.enc_i(x)).flatten(1)
        logits = model.head_i(feat)
    else:  # "imu"
        feat = model.backbone(model.enc_m(x)).flatten(1)
        logits = model.head_m(feat)
    return logits, y

def train_one_epoch_single(model, loader, async_mode, mod: str, lr=1e-3, criterion=None):
    model.train()
    ce = criterion or torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    total_loss, total_corr, total_n = 0.0, 0, 0
    for b in loader:
        logits, y = _single_logits_and_labels(model, b, async_mode, mod)
        loss = ce(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        total_loss += loss.item()
        total_corr += (logits.argmax(1) == y).sum().item()
        total_n    += y.size(0)
    return total_loss/ max(1,len(loader)), 100.0*total_corr/ max(1,total_n)

@torch.no_grad()
def eval_one_epoch_single(model, loader, async_mode, mod: str, criterion=None):
    model.eval()
    ce = criterion or torch.nn.CrossEntropyLoss()
    total_loss, total_corr, total_n = 0.0, 0, 0
    for b in loader:
        logits, y = _single_logits_and_labels(model, b, async_mode, mod)
        total_loss += ce(logits, y).item()
        total_corr += (logits.argmax(1) == y).sum().item()
        total_n    += y.size(0)
    return total_loss/ max(1,len(loader)), 100.0*total_corr/ max(1,total_n)

# ---------- epoch loops ----------
def train_one_epoch(model, loader, async_mode, criterions, optimizer, cagrad):
    model.train()
    Lw_fn, Li_fn, Lm_fn = criterions
    n = 0; loss_sum = np.zeros(3); acc_sum = np.zeros(3)
    for b in loader:
        (lw, li, lm), (yw, yi, ym) = forward_batch(model, b, async_mode)
        for t in (lw, li, lm):
            if not torch.isfinite(t).all():
                print("Non-finite logits detected:", {k: torch.isnan(t).any().item() for k in ['w','i','m']})
                break
        Lw, Li, Lm = Lw_fn(lw, yw), Li_fn(li, yi), Lm_fn(lm, ym) # reweigh loss with gcl
        step_cagrad_three(model, Lw, Li, Lm, optimizer, cagrad)
        with torch.no_grad():
            aw = (lw.argmax(1) == yw).float().mean().item() * 100
            ai = (li.argmax(1) == yi).float().mean().item() * 100
            am = (lm.argmax(1) == ym).float().mean().item() * 100
        loss_sum += np.array([Lw.item(), Li.item(), Lm.item()])
        acc_sum  += np.array([aw, ai, am]); n += 1
    return (loss_sum / max(1, n)).tolist(), (acc_sum / max(1, n)).tolist()

@torch.no_grad()
def eval_one_epoch(model, loader, async_mode, criterions):
    model.eval()
    Lw_fn, Li_fn, Lm_fn = criterions
    n = 0; loss_sum = np.zeros(3); acc_sum = np.zeros(3)

    # NEW for micro ensemble
    corr_sum = 0
    n_sum = 0

    for b in loader:
        (lw, li, lm), (yw, yi, ym) = forward_batch(model, b, async_mode)
        Lw, Li, Lm = Lw_fn(lw, yw), Li_fn(li, yi), Lm_fn(lm, ym)
        aw = (lw.argmax(1) == yw).float().mean().item() * 100
        ai = (li.argmax(1) == yi).float().mean().item() * 100
        am = (lm.argmax(1) == ym).float().mean().item() * 100

        if not async_mode:
            ps = F.softmax(lw, dim=1)
            pi = F.softmax(li, dim=1)
            pm = F.softmax(lm, dim=1)
            p  = (ps + pi + pm) / 3.0
            pred = p.argmax(1)
            corr_sum += (pred == yw).sum().item()
            n_sum    += yw.size(0)

        loss_sum += np.array([Lw.item(), Li.item(), Lm.item()])
        acc_sum  += np.array([aw, ai, am]); n += 1

    per_mod_loss = (loss_sum / max(1, n)).tolist()
    per_mod_acc  = (acc_sum  / max(1, n)).tolist()
    ens_acc = (100.0 * corr_sum / max(1, n_sum)) if not async_mode else None
    return per_mod_loss, per_mod_acc, ens_acc

# ----------- Helper for mask evaluation -----------
def _maybe_zero(x, use_flag):
    if use_flag: return x
    if x is None: return None
    return torch.zeros_like(x)

def forward_batch_masked(model, batch, async_mode: bool, use_w: bool, use_i: bool, use_m: bool):
    if async_mode:
        xw = _maybe_zero(batch["walkway"].to(DEVICE).float(), use_w)
        xi = _maybe_zero(batch["insole" ].to(DEVICE).float(), use_i)
        xm = _maybe_zero(batch["imu"    ].to(DEVICE).float(), use_m)
        yw = batch["y"]["walkway"].to(DEVICE).long()
        yi = batch["y"]["insole" ].to(DEVICE).long()
        ym = batch["y"]["imu"    ].to(DEVICE).long()
    else:
        xw0, xi0, xm0 = [t.to(DEVICE).float() for t in batch["xs"]]
        xw = _maybe_zero(xw0, use_w)
        xi = _maybe_zero(xi0, use_i)
        xm = _maybe_zero(xm0, use_m)
        y  = batch["y"].to(DEVICE).long()
        yw = yi = ym = y

    if getattr(model, "_is_taca3tri", False):
        f = lambda t: None if t is None else t.reshape(t.size(0), -1)
        lw, li, lm = model(f(xw), f(xi), f(xm), synced=(not async_mode))
        return (lw, li, lm), (yw, yi, ym)

    lw, li, lm = model(xw, xi, xm)
    return (lw, li, lm), (yw, yi, ym)

@torch.no_grad()
def eval_all_masks(model, loader, async_mode):
    out = {}
    for k, tup in MASK_COMBOS.items():
        out[k] = eval_with_mask(model, loader, async_mode, tup, verbose=True)
    return out

@torch.no_grad()
def eval_with_mask(model, loader, async_mode, mask, verbose=False):
    if isinstance(mask, str): mask = MASK_COMBOS[mask]
    use_w, use_i, use_m = map(bool, mask)
    model.eval()

    if not async_mode:
        corr_sum = n_sum = n_batches = 0
        for b in loader:
            (lw, li, lm), (y, _, _) = forward_batch_masked(model, b, False, use_w, use_i, use_m)
            probs = []
            if use_w: probs.append(F.softmax(lw, dim=1))
            if use_i: probs.append(F.softmax(li, dim=1))
            if use_m: probs.append(F.softmax(lm, dim=1))
            if not probs: continue
            p = sum(probs) / len(probs)
            pred = p.argmax(1)
            corr_sum += (pred == y).sum().item()
            n_sum    += y.size(0)
            n_batches += 1
        acc = 100.0 * corr_sum / max(1, n_sum)
        if verbose:
            enabled = "+".join([n for n,u in zip(("W","I","M"), (use_w,use_i,use_m)) if u]) or "None"
            print(f"[SYNC][mask={enabled}] acc={acc:5.2f}% over {n_batches} batches")
        return acc
    else:
        sum_aw = sum_ai = sum_am = 0.0; n_batches = 0
        for b in loader:
            (lw, li, lm), (yw, yi, ym) = forward_batch_masked(model, b, True, use_w, use_i, use_m)
            if use_w: sum_aw += (lw.argmax(1) == yw).float().mean().item() * 100.0
            if use_i: sum_ai += (li.argmax(1) == yi).float().mean().item() * 100.0
            if use_m: sum_am += (lm.argmax(1) == ym).float().mean().item() * 100.0
            n_batches += 1
        res = {}
        k = max(1, n_batches)
        if use_w: res["walkway"] = sum_aw / k
        if use_i: res["insole"]  = sum_ai / k
        if use_m: res["imu"]     = sum_am / k
        res["macro_enabled"] = sum(res.values()) / max(1, len(res)) if res else 0.0
        if verbose:
            enabled = "+".join([n for n,u in zip(("W","I","M"), (use_w,use_i,use_m)) if u]) or "None"
            print(f"[ASYNC][mask={enabled}] {res}")
        return res


# ---------- CV driver ----------
def make_loaders(prep, subj2label, args):
    mods = ("walkway","insole","imu")
    if args.async_loading:
        print("Using ASYNC data loading.")
        return make_async_loaders(
            prep, subj2label,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            modalities=mods
        ), True
    else:
        print("Using SYNC data loading.")
        return make_sync_loaders(
            prep, subj2label,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            modalities=mods
        ), False

def build_model(args, sync_flag: bool):
    if args.baseline is None:
        return WearGaitThreeModal(
            enc_out_ch=args.enc_out_ch,
            backbone_dim=args.backbone_dim,
            shared_out_ch=args.shared_out_ch,
            num_classes=args.num_classes,
            use_norm=args.use_norm,
            use_cosine=args.use_cosine,
            synchronized=sync_flag,
            pool_len=None,
        ).to(DEVICE)

    # DeepAVLite3 has a different init; pass only what it needs (use defaults for the rest)
    if args.baseline == "deepav_lite":
        return DeepAVLite3(
            num_classes=args.num_classes,
            synchronized=sync_flag
        ).to(DEVICE)
    if args.baseline == "focal":
        return FOCALSharedLatent3(
            # use defaults for dims unless you want to expose args for them:
            num_classes=args.num_classes,
            synchronized=sync_flag
        ).to(DEVICE)
    if args.baseline == "taca":
        # T = args.win_len; D: walkway=2, insole=13, imu=24
        model = TACA3TriWrapper(
            walk_T=args.win_len,  walk_D=2,
            insole_T=args.win_len, insole_D=13,
            imu_T=args.win_len,    imu_D=24,
            num_classes=args.num_classes,
            d_model=128, n_heads=4,
            n_tok_w=8, n_tok_i=8, n_tok_m=8,
            tau=1.0, gamma=1.5, schedule="const", dropout=0.1,
            use_time_shared=True,
            allow_async_cross=True,        
        ).to(DEVICE)
        # tiny flag so forward_batch knows to flatten & pass synced
        model._is_taca3tri = True
        return model

    if args.baseline == "shared_latent":
        return SharedLatent3(
            enc_out_ch=args.enc_out_ch,
            proj_ch=(getattr(args, "proj_ch", None) or args.enc_out_ch),  # safe fallback
            backbone_dim=args.backbone_dim,
            shared_out_ch=args.shared_out_ch,
            num_classes=args.num_classes,
            use_norm=args.use_norm,
            use_cosine=args.use_cosine,
            synchronized=sync_flag
        ).to(DEVICE)

    # other baselines keep using the common kwargs
    common = dict(
        enc_out_ch=args.enc_out_ch,
        backbone_dim=args.backbone_dim,
        shared_out_ch=args.shared_out_ch,
        num_classes=args.num_classes,
        synchronized=sync_flag
    )
    return {
        "early_fusion":  EarlyFusion3,
        "late_fusion":   LateFusion3,
        "cheap_xattn":   CheapXAttn3,
    }[args.baseline](**common).to(DEVICE)

def maybe_apply_drw(args, ep, tr_loader, async_mode, criterions):
    if args.single_mod or args.baseline or args.wm.lower() != "gcl":
        return
    counts = class_counts_per_mod(tr_loader, args.num_classes, async_mode)
    apply_drw_if_needed(ep, args, criterions, counts)


def run_cv(args):
    set_seed(args.seed)

    # subjects and folds
    pd_ids, hc_ids = discover_pd_hc(pd_path, hc_path)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, per_class=args.test_per_class, seed=args.seed)

    # accumulators across folds
    fold_macro = []
    fold_w, fold_i, fold_m = [], [], []

    mask_keys = list(MASK_COMBOS.keys())
    mask_fold_scores = {k: [] for k in mask_keys}

    for fi, (train_subs, test_subs) in enumerate(folds, 1):
        prep = prepare_split(train_subs, test_subs, data_dir=Path(output_dir),
                             win=args.win_len, hop=args.hop_len, modalities=("walkway","insole","imu"))
        (tr_loader, te_loader), async_mode = make_loaders(prep, subj2label, args)
        
        reseed_each = async_mode
        sync_flag  = (not async_mode)

        # --- model selection (respects --async_loading for all models) ---
        model = build_model(args, sync_flag)
        criterions, cagrad = build_criteria_and_cagrad(args, tr_loader, async_mode)
        single_crit = criterions[0] if args.single_mod else None
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

        is_single = args.single_mod is not None
        print(f"\n=== Fold {fi}/{len(folds)} ===")
        if is_single:
            print(f"→ Single-modality: {args.single_mod} (CE only)")
        
        best_macro, noimp = 0.0, 0
        # store best per-modality at the SAME epoch as best_macro
        best_w = best_i = best_m = 0.0
        best_state = None

        for ep in range(1, args.epochs + 1):
            if reseed_each and hasattr(tr_loader.dataset, "reseed"):
                tr_loader.dataset.reseed(args.seed + ep)

            maybe_apply_drw(args, ep, tr_loader, async_mode, criterions)
            
            if is_single:
                tl, ta = train_one_epoch_single(model, tr_loader, async_mode, args.single_mod, lr=args.lr, criterion=single_crit)
                vl, va = eval_one_epoch_single(model, te_loader, async_mode, args.single_mod, criterion=single_crit)
                improved = va > best_macro
                if improved:
                    best_macro = va
                    best_w = va if args.single_mod == "walkway" else 0.0
                    best_i = va if args.single_mod == "insole"  else 0.0
                    best_m = va if args.single_mod == "imu"     else 0.0
                noimp = 0 if improved else noimp + 1
                print(f"[Fold {fi}] Ep {ep:03d} | {args.single_mod} train {ta:5.2f}% L{tl:.3f} | val {va:5.2f}% best {best_macro:5.2f}%")
            else:
                (tlw, tli, tlm), (taw, tai, tam) = train_one_epoch(model, tr_loader, async_mode, criterions, optim, cagrad)
                (vlw, vli, vlm), (vaw, vai, vam), ens_acc = eval_one_epoch(model, te_loader, async_mode, criterions)

                macro = (vaw + vai + vam) / 3.0 if async_mode else ens_acc
                improved = macro > best_macro
                if improved:
                    best_macro = macro
                    best_w, best_i, best_m = vaw, vai, vam
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                noimp = 0 if improved else noimp + 1
                
                if async_mode:
                    print(f"[Fold {fi}] Ep {ep:03d} | L=[{tlw:.3f},{tli:.3f},{tlm:.3f}] acc=[{taw:5.1f},{tai:5.1f},{tam:5.1f}] | "
                          f"L=[{vlw:.3f},{vli:.3f},{vlm:.3f}] acc=[{vaw:5.1f},{vai:5.1f},{vam:5.1f}] | macro={macro:5.1f} best={best_macro:5.1f}")
                else:
                    print(f"[Fold {fi}] Ep {ep:03d} | L=[{tlw:.3f},{tli:.3f},{tlm:.3f}] acc=[{taw:5.1f},{tai:5.1f},{tam:5.1f}] | "
                          f"L=[{vlw:.3f},{vli:.3f},{vlm:.3f}] acc=[{vaw:5.1f},{vai:5.1f},{vam:5.1f}] | ens={ens_acc:5.1f} best={best_macro:5.1f}")

            if noimp >= args.patience:
                print(f"[Fold {fi}] Early stop at epoch {ep}")
                break

        # --- restore best and run masked evals (skip if single_mod) ---
        if (not is_single) and best_state is not None:
            model.load_state_dict(best_state, strict=True)
            # compute and record best-per-mask for this fold
            for mk in mask_keys:
                r = eval_with_mask(model, te_loader, async_mode, mk, verbose=True)
                if async_mode:
                    score = float(r["macro_enabled"])
                else:
                    score = float(r)  # ensemble under enabled streams
                mask_fold_scores[mk].append(score)

        print(f"[Fold {fi}] Best macro acc: {best_macro:.2f}% "
              f"(W={best_w:.2f} I={best_i:.2f} M={best_m:.2f})")
        fold_macro.append(best_macro); fold_w.append(best_w); fold_i.append(best_i); fold_m.append(best_m)

    # ---- summary ----
    macro_mean, macro_std = float(np.mean(fold_macro)), float(np.std(fold_macro))
    w_mean, w_std = float(np.mean(fold_w)), float(np.std(fold_w))
    i_mean, i_std = float(np.mean(fold_i)), float(np.std(fold_i))
    m_mean, m_std = float(np.mean(fold_m)), float(np.std(fold_m))

    print("\n=== Summary ===")
    print(f"Macro acc mean ± std: {macro_mean:.2f}% ± {macro_std:.2f}%")
    print(f"Per-mod acc mean ± std: "
          f"[walkway {w_mean:.2f} ± {w_std:.2f}]  "
          f"[insole {i_mean:.2f} ± {i_std:.2f}]  "
          f"[imu {m_mean:.2f} ± {m_std:.2f}]")

    if mask_fold_scores and all(len(v)>0 for v in mask_fold_scores.values()):
        print("\n=== Masked accuracy at best epoch (avg across folds) ===")
        for mk in mask_keys:
            arr = np.array(mask_fold_scores[mk], dtype=float)
            print(f"[{mk:5}] {arr.mean():5.2f}% ± {arr.std():4.2f}%  over {len(arr)} folds")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # CLI

    # folds + windows
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--test_per_class", type=int, default=8)
    ap.add_argument("--win_len", type=int, default=64)
    ap.add_argument("--hop_len", type=int, default=64)  # non-overlap if == win_len

    # training
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--async_loading", action="store_true")
    ap.add_argument("--single_mod", type=str, choices=["walkway","insole","imu"], default=None)
    ap.add_argument("--proj_ch", type=int, default=16)

    # model
    ap.add_argument("--enc_out_ch", type=int, default=12)
    ap.add_argument("--backbone_dim", type=int, default=8)
    ap.add_argument("--shared_out_ch", type=int, default=16)
    ap.add_argument("--use_norm", action="store_true")
    ap.add_argument("--use_cosine", action="store_true")
    ap.add_argument("--baseline", type=str, default=None, 
                    choices=["early_fusion","late_fusion","shared_latent",
                             "cheap_xattn", "deepav_lite", "focal", "taca"])

    # losses / weighting
    ap.add_argument("--wm", type=str, default="gcl", choices=["ce", "class_wt", "gcl"])
    ap.add_argument("--gcl_m", type=float, default=0.2)
    ap.add_argument("--gcl_s", type=float, default=25)
    ap.add_argument("--noise_mul", type=float, default=0.0)
    ap.add_argument("--drw_warmup", type=int, default=0)

    # CAGrad
    ap.add_argument("--alpha", type=float, default=0.5, help="CAGrad c; 0 disables CAGrad")

    args = ap.parse_args()
    run_cv(args)

"""
SEEDS=(2 3 4 42 43 44)
for s in "${SEEDS[@]}"; do
  nohup python -u /home/yongjie/Minlin/CIKM-2025-Minlin/train/weargait_train.py \
    --seed "$s" \
    --baseline taca       --wm class_wt \
    > "/home/yongjie/Minlin/CIKM-2025-Minlin/data/WearGait/logs/inverse_freq/sync_taca_wt_seed${s}.out" 2>&1 &
  sleep 0.2
done

SEEDS=(2 3 4 42 43 44)
for s in "${SEEDS[@]}"; do
  nohup python -u /home/yongjie/Minlin/CIKM-2025-Minlin/train/weargait_train.py \
    --seed "$s" \
    --baseline deepav_lite       --wm class_wt \
    --async_loading \
    > "/home/yongjie/Minlin/CIKM-2025-Minlin/data/WearGait/logs/inverse_freq/async_deepav_wt_seed${s}.out" 2>&1 &
  sleep 0.2
done

"""
