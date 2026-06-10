# data_processing/dataloader_weargait.py
"""
Load per-subject PKLs and:
1) fit z-score stats on TRAIN ONLY (insole+IMU),
2) cut into NON-OVERLAPPING windows (strict full windows),
3) build sync/async datasets and dataloaders with SEPARATE modality tensors.

Public API:
- discover_subjects(data_dir)
- build_subj2label(pd_ids, hc_ids)
- make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds, per_class, seed)
- prepare_split(train_subs, test_subs, *, data_dir, win, hop, modalities)
- make_sync_loaders(prep, subj2label, batch_size, num_workers, seed, modalities)
- make_async_loaders(prep, subj2label, batch_size, num_workers, seed, modalities)
- save_stats(stats, path)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data/WearGait/WearGait_preproc_SPmT_30Hz")
MIN_STD      = 1e-6
IMU_SITES    = ["L_Ankle","R_Ankle","L_DorsalFoot","R_DorsalFoot","L_MidLatThigh","R_MidLatThigh","L_LatShank","R_LatShank"]
DEFAULT_MODALITIES = ("walkway","insole","imu")
# channel sets
INSOLE_NUMERIC = [
    "LTotalForce_BW","RTotalForce_BW","SumForce_BW",
    "LCoP_X","LCoP_Y","RCoP_X","RCoP_Y",
    "Linsole_Acc_X","Linsole_Acc_Y","Linsole_Acc_Z",
    "Rinsole_Acc_X","Rinsole_Acc_Y","Rinsole_Acc_Z",
]
WALKWAY_NUMERIC = ["L Foot Pressure_BW","R Foot Pressure_BW"]

IMU_AXES = ("E","N","U")
IMU_FIXED = [f"{s}_FreeAcc_{ax}" for s in IMU_SITES for ax in IMU_AXES]  # 8*3 = 24
INSOLE_FIXED = [
    "LTotalForce_BW","RTotalForce_BW","SumForce_BW",
    "LCoP_X","LCoP_Y","RCoP_X","RCoP_Y",
    "Linsole_Acc_X","Linsole_Acc_Y","Linsole_Acc_Z",
    "Rinsole_Acc_X","Rinsole_Acc_Y","Rinsole_Acc_Z",
]
WALKWAY_FIXED = ["L Foot Pressure_BW","R Foot Pressure_BW"]

# ─── Subject discovery ───────────────────────────────────────────────────────
def discover_subjects(data_dir: Path = DEFAULT_DATA_DIR) -> List[str]:
    """List subjects by scanning walkway PKLs."""
    subs = {p.name.split("_")[0] for p in Path(data_dir).glob("*_walkway.pkl")}
    return sorted(subs)

def build_subj2label(pd_ids: List[str], hc_ids: List[str]) -> Dict[str, int]:
    """Map subject id → class index (PD=1, HC=0 by convention)."""
    return {**{s:1 for s in pd_ids}, **{s:0 for s in hc_ids}}

def make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=10, per_class=8, seed=0):
    """Disjoint test sets per fold. Each test set: per_class PD + per_class HC."""
    assert len(pd_ids) >= n_folds*per_class and len(hc_ids) >= n_folds*per_class, "Not enough subjects."
    rng = random.Random(seed)
    pd_pool = pd_ids[:] ; hc_pool = hc_ids[:]
    rng.shuffle(pd_pool); rng.shuffle(hc_pool)
    used_pd = pd_pool[:n_folds*per_class]; used_hc = hc_pool[:n_folds*per_class]
    folds = []
    for f in range(n_folds):
        te_pd = sorted(used_pd[f*per_class:(f+1)*per_class])
        te_hc = sorted(used_hc[f*per_class:(f+1)*per_class])
        te = te_pd + te_hc
        tr = sorted([s for s in (pd_ids + hc_ids) if s not in te])
        folds.append((tr, te))
    return folds

def ensure_cols(df: pd.DataFrame,
                required_cols: list[str],
                stats: dict[str, tuple[float, float]] | None = None,
                pre_norm: bool = False) -> pd.DataFrame:
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            fill = (stats[c][0] if (pre_norm and stats is not None and c in stats) else 0.0)
            out[c] = fill
        else:
            # If the column exists but is all non-finite → fill with mean (or 0)
            col = pd.to_numeric(out[c], errors="coerce")
            if not np.isfinite(col).any():
                fill = (stats[c][0] if (pre_norm and stats is not None and c in stats) else 0.0)
                out[c] = fill
    return out[required_cols]

def _report_nan_health(df: pd.DataFrame, sid: str, modality: str,
                       required_cols: list[str], thresh: float = 0.5) -> None:
    """
    Print a warning if this subject/modality has too many invalid values.
    - 'all-NaN columns' ratio >= thresh  → warn
    - overall non-finite cell ratio > 0.5 → warn
    Only checks columns that are present in `df` and listed in required_cols.
    """
    if df.empty:
        print(f"[WARN] {sid:<8} | {modality:<7} | DataFrame is EMPTY")
        return

    cols = [c for c in required_cols if c in df.columns]
    if not cols:
        print(f"[WARN] {sid:<8} | {modality:<7} | No required numeric columns present")
        return

    # use numeric view and coerce weird types
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(X)

    # 1) columns that are entirely non-finite
    all_nan_cols = (~finite_mask).all(axis=0)
    n_all_nan = int(all_nan_cols.sum())
    ratio_all_nan_cols = n_all_nan / len(cols)

    # 2) overall non-finite ratio across all cells
    overall_nonfinite_ratio = 1.0 - (finite_mask.sum() / finite_mask.size)

    if ratio_all_nan_cols >= thresh or n_all_nan == len(cols):
        print(f"[WARN] {sid:<8} | {modality:<7} | "
              f"{n_all_nan}/{len(cols)} columns ALL-NaN "
              f"({ratio_all_nan_cols:.2f})")

    if overall_nonfinite_ratio > 0.5:
        print(f"[WARN] {sid:<8} | {modality:<7} | "
              f"overall non-finite ratio {overall_nonfinite_ratio:.2f} (>0.50)")

MIN_STD = 1e-6  # keep your constant

# ─── I/O ─────────────────────────────────────────────────────────────────────
def _load_df(data_dir: Path, sid: str, suffix: str) -> pd.DataFrame:
    """Load a pickle DataFrame for a subject and suffix."""
    p = Path(data_dir) / f"{sid.lower()}_{suffix}"
    return pd.read_pickle(p) if p.exists() else pd.DataFrame()

def load_subject_streams(data_dir: Path, sid: str) -> Dict[str, pd.DataFrame]:
    """Read per-subject PKLs. Assumes `_base` suffix already removed."""
    return {
        "walkway": _load_df(data_dir, sid, "walkway.pkl"),
        "insole":  _load_df(data_dir, sid, "insole.pkl"),
        "imu":     _load_df(data_dir, sid, "imu.pkl"),
    }

# ─── Tuple expansion helpers ─────────────────────────────────────────────────
def _expand_tuple_cols(df: pd.DataFrame, col: str, out_prefix: str, axes=("X","Y","Z")):
    if col not in df.columns: return df
    arr = np.vstack(df[col].astype(object).apply(lambda t: np.asarray(t, dtype=float)).to_numpy())
    for i, ax in enumerate(axes): df[f"{out_prefix}_{ax}"] = arr[:, i]
    return df.drop(columns=[col])

def expand_insole(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return ensure_cols(pd.DataFrame(), INSOLE_FIXED)
    df = df.copy()
    df = _expand_tuple_cols(df, "Linsole_Acc", "Linsole_Acc", ("X","Y","Z"))
    df = _expand_tuple_cols(df, "Rinsole_Acc", "Rinsole_Acc", ("X","Y","Z"))
    return ensure_cols(df, INSOLE_FIXED)

def expand_imu(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return ensure_cols(pd.DataFrame(), IMU_FIXED)
    df = df.copy()
    for s in IMU_SITES:
        c = f"{s}_FreeAcc"
        if c in df.columns:
            df = _expand_tuple_cols(df, c, c, ("E","N","U"))
    return ensure_cols(df, IMU_FIXED)

def imu_numeric_cols(df_cols) -> List[str]:
    out = []
    for s in IMU_SITES:
        for ax in ("E","N","U"):
            c = f"{s}_FreeAcc_{ax}"
            if c in df_cols: out.append(c)
    return out

# ─── Z-score on TRAIN only (insole + IMU) ────────────────────────────────────
def fit_stats_on_train(train_subjects: List[str], data_dir: Path) -> Dict[str, Tuple[float,float]]:
    sums, sumsqs, counts = {}, {}, {}
    def _accumulate(vals: np.ndarray, cols: List[str]):
        for i, c in enumerate(cols):
            x = vals[:, i]
            m = np.isfinite(x)
            if not m.any(): continue
            x = x[m].astype(float)
            sums[c]   = sums.get(c,0.0)   + float(x.sum())
            sumsqs[c] = sumsqs.get(c,0.0) + float(np.dot(x,x))
            counts[c] = counts.get(c,0)   + int(x.size)

    for sid in train_subjects:
        st = load_subject_streams(data_dir, sid)
        di = expand_insole(st["insole"])
        if not di.empty:
            cols = [c for c in INSOLE_NUMERIC if c in di.columns]
            if cols: _accumulate(di[cols].to_numpy(dtype=float), cols)
        dm = expand_imu(st["imu"])
        if not dm.empty:
            cols = imu_numeric_cols(dm.columns)
            if cols: _accumulate(dm[cols].to_numpy(dtype=float), cols)

    stats = {}
    for c, n in counts.items():
        mean = sums[c]/n
        var  = max((sumsqs[c]/n) - mean**2, 0.0)
        std  = max(np.sqrt(var), MIN_STD)
        stats[c] = (mean, std)
    return stats

def apply_stats(df: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, (m, s) in stats.items():
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)

        # replace any non-finite with the train mean before z-score
        x[~np.isfinite(x)] = m if np.isfinite(m) else 0.0
        s_eff = s if (np.isfinite(s) and s > MIN_STD) else MIN_STD

        z = (x - (m if np.isfinite(m) else 0.0)) / s_eff
        # final guard: remove any residual NaN/Inf from weird inputs
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        out[c] = z
    return out

# ─── Windowing (strict full windows) ─────────────────────────────────────────
def window_indices(n_frames: int, win: int, hop: int) -> List[Tuple[int,int,int]]:
    idx = []
    if n_frames <= 0 or n_frames < win: return idx
    w = 0; i = 0
    while w + win <= n_frames:
        idx.append((i, w, w+win))
        w += hop; i += 1
    return idx

def build_windows_per_subject(
    data_dir: Path,
    sid: str,
    stats: Dict[str, Tuple[float,float]],
    win: int,
    hop: int
) -> Dict[str, Dict[str, np.ndarray]]:
    out = {"walkway": {}, "insole": {}, "imu": {}}
    st = load_subject_streams(data_dir, sid)

    # walkway (usually complete; keep as-is)
    dw = st["walkway"]
    # _report_nan_health(dw, sid, "walkway", WALKWAY_FIXED) 
    Xw = ensure_cols(dw, WALKWAY_FIXED).to_numpy(dtype=float)
    for wid, s0, s1 in window_indices(len(Xw), win, hop):
        out["walkway"][f"{sid}|walkway|{wid}"] = Xw[s0:s1]

    # insole: mean-fill missing BEFORE z-score
    di_raw = expand_insole(st["insole"])
    # _report_nan_health(di_raw, sid, "insole", INSOLE_FIXED)
    di_filled = ensure_cols(di_raw, INSOLE_FIXED, stats=stats, pre_norm=True)
    di = apply_stats(di_filled, stats)
    Xi = di.to_numpy(dtype=float)
    for wid, s0, s1 in window_indices(len(Xi), win, hop):
        out["insole"][f"{sid}|insole|{wid}"] = Xi[s0:s1]

    # imu: mean-fill missing BEFORE z-score
    dm_raw = expand_imu(st["imu"])
    # _report_nan_health(dm_raw, sid, "imu", IMU_FIXED)
    dm_filled = ensure_cols(dm_raw, IMU_FIXED, stats=stats, pre_norm=True)
    dm = apply_stats(dm_filled, stats)
    Xm = dm.to_numpy(dtype=float)
    for wid, s0, s1 in window_indices(len(Xm), win, hop):
        out["imu"][f"{sid}|imu|{wid}"] = Xm[s0:s1]

    return out


# ─── Sync/Async key maps ─────────────────────────────────────────────────────
def _build_index_maps(
    per_subj: Dict[str, Dict[str, Dict[str,np.ndarray]]],
    modalities: Tuple[str,...]
) -> Tuple[List[str], List[Tuple[str,...]]]:
    async_keys = []
    for m in modalities:
        for sid in per_subj:
            async_keys += sorted(per_subj[sid][m].keys())

    sync_pairs = []
    for sid in per_subj:
        sets = [ {k.split("|")[-1] for k in per_subj[sid][m].keys()} for m in modalities ]
        if not all(sets): continue
        common = sorted(set.intersection(*sets), key=lambda x:int(x))
        for wid in common:
            tup = tuple(f"{sid}|{m}|{wid}" for m in modalities)
            ok = True
            for i, k in enumerate(tup):
                if k not in per_subj[sid][modalities[i]]:
                    ok = False; break
            if ok: sync_pairs.append(tup)
    return async_keys, sync_pairs

# ─── Datasets + collate ─────────────────────────────────────────────────────
def _subj_from_key(k: str) -> str:
    return k.split("|", 1)[0]

class WearGaitMultiAsyncDataset(Dataset):
    """
    Async triplets without replacement.
    Epoch length = min(len(walkway), len(insole), len(imu)).
    Returns per-modality labels: batch["y"]["walkway"|"insole"|"imu"].
    """
    def __init__(self, stores: Dict[str, Dict[str, np.ndarray]],
                 modalities: Tuple[str,...], subj2label: Dict[str,int], seed: int = 0):
        self.modalities  = modalities
        self.stores      = stores
        self.subj2label  = subj2label
        self._rng        = random.Random(seed)

        self._keys_full  = {m: sorted(stores[m].keys()) for m in modalities}
        self._lens_full  = {m: len(self._keys_full[m]) for m in modalities}
        self._min_len    = min(self._lens_full.values())

        # per-modality permutations (no replacement within an epoch)
        self._perms = {}
        for m in modalities:
            idxs = list(range(self._lens_full[m]))
            self._rng.shuffle(idxs)
            self._perms[m] = idxs[:self._min_len]

    def reseed(self, seed: int):
        self._rng = random.Random(seed)
        for m in self.modalities:
            idxs = list(range(self._lens_full[m]))
            self._rng.shuffle(idxs)
            self._perms[m] = idxs[:self._min_len]

    def __len__(self): 
        return self._min_len

    def __getitem__(self, idx):
        out = {"keys": {}, "y": {}}
        for m in self.modalities:
            k = self._keys_full[m][ self._perms[m][idx] ]   # "SID|mod|wid"
            x = self.stores[m][k].astype(np.float32)
            sid = k.split("|", 1)[0]
            out[m] = torch.from_numpy(x)                    # (T,D)
            out["keys"][m] = k
            out["y"][m] = torch.tensor(self.subj2label[sid], dtype=torch.long)
        return out


class WearGaitSyncDataset(Dataset):
    """Return aligned modality windows per item. Keeps streams separate."""
    def __init__(self, stores: Tuple[Dict[str,np.ndarray], ...], pairs: List[Tuple[str,...]],
                 subj2label: Dict[str,int]):
        self.stores = stores
        self.pairs  = pairs
        self.subj2label = subj2label
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ks = self.pairs[i]
        xs = [ torch.from_numpy(self.stores[j][ks[j]].astype(np.float32)) for j in range(len(self.stores)) ]
        y  = torch.tensor(self.subj2label[_subj_from_key(ks[0])], dtype=torch.long)
        return {"xs": xs, "keys": ks, "y": y}

def _collate_sync(batch: List[Dict[str,Any]]):
    xs_by_mod = list(zip(*[b["xs"] for b in batch]))       # list of lists by modality
    out = {
        "xs":  [torch.stack(mod_list, dim=0) for mod_list in xs_by_mod],  # each (B,T,D)
        "keys": [b["keys"] for b in batch],
        "y": torch.stack([b["y"] for b in batch], dim=0)                  # (B,)
    }
    return out

def _make_collate_async(modalities):
    def _collate(batch):
        B = len(batch)
        out = {"keys": {m: [] for m in modalities}, "y": {}}
        for m in modalities:
            xs = [b[m] for b in batch]                          # list of (T,D)
            out[m] = torch.stack(xs, 0)                         # (B,T,D)
            out["keys"][m] = [b["keys"][m] for b in batch]
            out["y"][m] = torch.stack([b["y"][m] for b in batch], 0)  # (B,)
        return out
    return _collate


# ─── Public pipeline helpers ─────────────────────────────────────────────────
def prepare_split(
    train_subs: List[str],
    test_subs:  List[str],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    win: int = 64,
    hop: int = 64,
    modalities: Tuple[str,...] = DEFAULT_MODALITIES
):
    """Fit stats on train only, window train+test, build stores and sync indices."""
    stats = fit_stats_on_train(train_subs, data_dir)

    per_subj_train = {sid: build_windows_per_subject(data_dir, sid, stats, win, hop) for sid in train_subs}
    per_subj_test  = {sid: build_windows_per_subject(data_dir, sid, stats, win, hop) for sid in test_subs}

    train_stores = {m:{} for m in modalities}
    test_stores  = {m:{} for m in modalities}
    for sid, dd in per_subj_train.items():
        for m in modalities: train_stores[m].update(dd[m])
    for sid, dd in per_subj_test.items():
        for m in modalities: test_stores[m].update(dd[m])

    _, train_sync = _build_index_maps(per_subj_train, modalities)
    _, test_sync  = _build_index_maps(per_subj_test,  modalities)

    return {
        "train_subs": train_subs, "test_subs": test_subs,
        "stats": stats,
        "train_stores": train_stores, "test_stores": test_stores,
        "train_sync": train_sync, "test_sync": test_sync
    }

def make_sync_loaders(
    prep: Dict[str,Any],
    subj2label: Dict[str,int],
    *,
    batch_size=64, num_workers=4, seed=0,
    modalities: Tuple[str,...] = DEFAULT_MODALITIES
) -> Tuple[DataLoader,DataLoader]:
    
    g = torch.Generator().manual_seed(seed)
    train_ds = WearGaitSyncDataset(tuple(prep["train_stores"][m] for m in modalities), prep["train_sync"], subj2label=subj2label)
    test_ds  = WearGaitSyncDataset(tuple(prep["test_stores"][m]  for m in modalities), prep["test_sync"],  subj2label=subj2label)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, generator=g, collate_fn=_collate_sync)
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, generator=g, collate_fn=_collate_sync)
    return tr, te

# --- update make_async_loaders to pass seed ----------------------------------
def make_async_loaders(
    prep: Dict[str,Any],
    subj2label: Dict[str,int],
    *,
    batch_size=64, num_workers=4, seed=0,
    modalities: Tuple[str,...] = DEFAULT_MODALITIES
) -> Tuple[DataLoader,DataLoader]:
    
    g = torch.Generator().manual_seed(seed)
    collate = _make_collate_async(modalities)

    train_ds = WearGaitMultiAsyncDataset(prep["train_stores"], modalities, subj2label=subj2label, seed=seed)
    test_ds  = WearGaitMultiAsyncDataset(prep["test_stores"],  modalities, subj2label=subj2label,  seed=seed+1)

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                    pin_memory=True, generator=g, collate_fn=collate)
    te = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, generator=g, collate_fn=collate)
    return tr, te


# ─── Persistence ─────────────────────────────────────────────────────────────
def save_stats(stats: Dict[str,Tuple[float,float]], path: Path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
