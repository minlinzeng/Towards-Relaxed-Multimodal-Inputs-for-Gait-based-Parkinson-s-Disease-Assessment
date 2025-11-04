# data/efficient_dataloader.py

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SKELETON_LEN = 101
DEFAULT_SENSOR_LEN   = 65
NUM_CLASSES          = 3
MIN_STD              = 1e-4

# ─── Helpers ─────────────────────────────────────────────────────────────────

def pad_or_trim(
    seq: np.ndarray,
    target_len: int,
    pad_value: float = 0.0) -> np.ndarray:
    """Pad (at end) or trim a temporal sequence to exactly target_len frames."""
    L = seq.shape[0]
    if L == target_len:
        return seq
    elif L > target_len:
        return seq[:target_len]
    else:
        pad_len = target_len - L
        pad = np.full((pad_len, *seq.shape[1:]), pad_value, dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)

def compute_class_weights(counts: List[int]) -> torch.Tensor:
    """Invert frequencies and normalize so weights.sum()==NUM_CLASSES."""
    t = torch.tensor(counts, dtype=torch.float32)
    w = 1.0 / (t + 1e-8)
    return (w / w.sum() * NUM_CLASSES)

def group_by_subject(keys: List[str]) -> Dict[str, List[str]]:
    """Map SUBID → [all keys starting with that SUBID]."""
    out = defaultdict(list)
    for k in keys:
        sub = k.split("_")[0]
        out[sub].append(k)
    return out

def build_synced_pairs(
    pose_map: Dict[str,List[str]],
    sens_map: Dict[str,List[str]]) -> List[Tuple[str,str]]:
    """
    For each subject, align pose/sensor by matching their last two segments.
    Returns list of (pose_key, sensor_key).
    """
    pairs = []
    for sub, pkeys in pose_map.items():
        skeys = sens_map.get(sub, [])
        # group sensor by segment suffix
        seg_dict = defaultdict(list)
        for sk in skeys:
            seg = "_".join(sk.split("_")[-2:])
            seg_dict[seg].append(sk)
        # match poses
        for pk in pkeys:
            seg = "_".join(pk.split("_")[-2:])
            for sk in seg_dict.get(seg, []):
                pairs.append((pk, sk))
    return pairs

def oversample_equally(
    pairs: List[Tuple[str,str]],
    get_label: callable) -> List[Tuple[str,str]]:
    """
    Balanced oversampling: ensure each class appears equally often.
    """
    cls2pairs = defaultdict(list)
    for pk, sk in pairs:
        cls2pairs[get_label(pk)].append((pk, sk))
    max_n = max(len(v) for v in cls2pairs.values())
    balanced = []
    for c, group in cls2pairs.items():
        for _ in range(max_n):
            balanced.append(random.choice(group))
    random.shuffle(balanced)
    return balanced

# ─── Walk-specific preprocessing ──────────────────────────────────────────────
def center_poses(pose_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Subtract root joint so every pose is pelvis-centered."""
    ROOT = 0
    out = {}
    for k, arr in pose_dict.items():
        out[k] = arr - arr[:, ROOT:ROOT+1, :]
    return out

def normalize_poses(pose_dict: Dict[str, np.ndarray], method: str = 'minmax') -> Dict[str, np.ndarray]:
    """
    Per-video normalization:
      - 'minmax': scale each video into [0,1]
      - 'zscore': global z-score across all videos
    """
    if method == 'minmax':
        out = {}
        for k, arr in pose_dict.items():
            mins = arr.min(axis=(0,1))
            maxs = arr.max(axis=(0,1))
            out[k] = (arr - mins) / (maxs - mins + 1e-6)
        return out
    elif method == 'zscore':
        all_frames = np.vstack(list(pose_dict.values()))
        mean = all_frames.mean(axis=0)
        std  = all_frames.std(axis=0)
        std[std < MIN_STD] = 1.0
        return {k: (arr - mean) / std for k, arr in pose_dict.items()}
    else:
        return pose_dict

# ─── Dataset Classes ─────────────────────────────────────────────────────────
class SkeletonDataset(Dataset):
    def __init__(
        self,
        pose_dict: Dict[str,np.ndarray],
        selected_subjects: List[str],
        pad_length: int = DEFAULT_SKELETON_LEN,
    ):
        if selected_subjects is None:
            keys = list(pose_dict.keys())
        else:
            keys = [k for k in pose_dict if any(k.startswith(ss) for ss in selected_subjects)]
        self.poses = {k: pad_or_trim(pose_dict[k], pad_length) for k in keys}
        self.keys = list(self.poses.keys())
        self.pad_length = pad_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.poses[key].astype(np.float32)
        return data, key

class SensorDataset(Dataset):
    def __init__(
        self,
        sensor_dict: Dict[str,np.ndarray],
        selected_subjects: List[str],
        pad_length: int = DEFAULT_SENSOR_LEN,
    ):
        if not selected_subjects:
            keys = list(sensor_dict.keys())
        else:
            keys = [k for k in sensor_dict if any(k.startswith(ss) for ss in selected_subjects)]
        self.sensors = {k: pad_or_trim(sensor_dict[k], pad_length) for k in keys}
        self.keys = list(self.sensors.keys())
        self.pad_length = pad_length

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.sensors[key].astype(np.float32)
        return data, key

class FusionDataset(Dataset):
    """
    Multimodal dataset that can sample asynchronously or in time‐synchronized pairs.
    """
    def __init__(
        self,
        pose_dict: Dict[str,np.ndarray],
        sensor_dict: Dict[str,np.ndarray],
        subject_label_map: Dict[str, int] = None,
        pose_label_map:    Dict[str, int] = None,
        sensor_label_map:  Dict[str, int] = None,
        selected_subjects: List[str] = None,
        synchronized:      bool = False,
        seed:              int = 0,
        pad_skel:          int = DEFAULT_SKELETON_LEN,
        pad_sens:          int = DEFAULT_SENSOR_LEN):
        
        self.pose_ds            = SkeletonDataset(pose_dict, selected_subjects, pad_skel)
        self.sens_ds            = SensorDataset(sensor_dict, selected_subjects, pad_sens)
        self.synchronized       = synchronized
        self.subject_label_map  = subject_label_map
        self.pose_label_map     = pose_label_map
        self.sensor_label_map   = sensor_label_map

        # if time-sync requested, build (pose_key, sensor_key) pairs
        if self.synchronized:
            pose_map = group_by_subject(self.pose_ds.keys)
            sens_map = group_by_subject(self.sens_ds.keys)
            pairs    = build_synced_pairs(pose_map, sens_map)

            if seed is not None:
                random.seed(seed)
                if self.pose_label_map is not None:
                    # for walk: Pose keys like "SUBxx_on_walk_..."; labels map uses "SUBxx_on"
                    getter = lambda pk: self.pose_label_map["_".join(pk.split("_")[:2])]
                else:
                    getter = lambda pk: self.subject_label_map[pk.split("_")[0]]
                pairs = oversample_equally(pairs, getter)
            self.pairs = pairs

    def __len__(self):
        if self.synchronized:
            return len(self.pairs)
        return max(len(self.pose_ds), len(self.sens_ds))

    def __getitem__(self, idx):
        if self.synchronized:
            pk, sk = self.pairs[idx]
            xs     = self.pose_ds.poses[pk].astype(np.float32)
            xt     = self.sens_ds.sensors[sk].astype(np.float32)

            # Pose label: map "SUBxx_on" → 0..NUM_CLASSES-1
            pose_key = "_".join(pk.split("_")[:2])
            if self.pose_label_map is not None:
                ys = self.pose_label_map[pose_key]
            else:
                ys = self.subject_label_map[pk.split("_")[0]]

            # Sensor label: full segment key lookup
            if self.sensor_label_map is not None:
                yt = self.sensor_label_map[sk]
            else:
                yt = self.subject_label_map[sk.split("_")[0]]

        else:
            # asynchronous sampling: wrap-around each modality independently
            i = idx % len(self.pose_ds)
            j = idx % len(self.sens_ds)
            xs, pk = self.pose_ds[i]
            xt, sk = self.sens_ds[j]

            pose_key = "_".join(pk.split("_")[:2])
            if self.pose_label_map is not None:
                ys = self.pose_label_map[pose_key]
            else:
                ys = self.subject_label_map[pk.split("_")[0]]

            if self.sensor_label_map is not None:
                yt = self.sensor_label_map[sk]
            else:
                yt = self.subject_label_map[sk.split("_")[0]]

        return {
            "skeleton":       torch.from_numpy(xs),           # (T_s, D_s)
            "sensor":         torch.from_numpy(xt),           # (T_g, D_g)
            "label_skeleton": torch.tensor(ys, dtype=torch.long),
            "label_sensor":   torch.tensor(yt, dtype=torch.long)
        }

# ─── DataLoader Factory ───────────────────────────────────────────────────────
def seed_worker(worker_id: int):
    """
    DataLoader worker init function to ensure reproducible shuffling
    when using multiple workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_fusion_loaders(
    dataset:        str,
    reader:         Any,                          
    train_subjects: List[str],
    eval_subjects:  List[str],
    batch_size:     int  = 32,
    synchronized:   bool = False,
    seed:           int  = 0,
    num_workers:    int  = 4,
    pad_skel:       int  = DEFAULT_SKELETON_LEN,
    pad_sens:       int  = DEFAULT_SENSOR_LEN,
    modality:       str  = "multimodal"
) -> Tuple[DataLoader,DataLoader]:
    """
    Builds train & eval DataLoaders of (optionally) time-aligned multimodal data.
    """
    random.seed(seed)
    # 1) raw dicts & labels
    if dataset == "walk":
        train_subs       = train_subjects[:]
        eval_subs        = eval_subjects[:]
        pose_dict        = reader.pose_dict.copy()
        sensor_dict      = reader.sensor_dict.copy()
        subject_label_map= None
        pose_label_map   = reader.pose_label_dict.copy()
        sensor_label_map = reader.sensor_label_dict.copy()

        # center & normalize skeletons
        pose_dict = center_poses(pose_dict)
        pose_dict = normalize_poses(pose_dict, method='minmax')

        # split GRF into per-trial entries
        new_sdict, new_smap = {}, {}
        for key, arr in sensor_dict.items():
            if arr.ndim == 3:
                for i in range(arr.shape[1]):
                    seg = f"{key}_{i}"
                    new_sdict[seg] = pad_or_trim(arr[:, i, :], pad_sens)
                    new_smap[seg]  = sensor_label_map[key]
            else:
                new_sdict[key] = pad_or_trim(arr, pad_sens)
                new_smap[key]  = sensor_label_map[key]
        sensor_dict      = new_sdict
        sensor_label_map = new_smap

    else:  # "turn"
        train_subs        = train_subjects[:]
        eval_subs         = eval_subjects[:]
        pose_dict         = reader.pose_dict
        sensor_dict       = reader.sensor_dict
        subject_label_map = {
            subj: (lbls[0] if isinstance(lbls, (list,tuple)) else int(lbls))
            for subj, lbls in reader.labels_dict.items()
            if subj not in ("SUB10","SUB30","SUB22")
        }
        pose_label_map    = None
        sensor_label_map  = None

    # 2) filter subjects by requested modality
    #    derive two-token prefixes present in the final dicts
    if dataset == "walk":
        pose_pfx = { "_".join(k.split("_")[:2]) for k in pose_dict.keys() }
        sens_pfx = { "_".join(k.split("_")[:2]) for k in sensor_dict.keys() }
        def has_data(subj: str) -> bool:
            if modality == "skeleton":
                return subj in pose_pfx
            elif modality == "sensor":
                return subj in sens_pfx
            else:  # multimodal
                return (subj in pose_pfx) or (subj in sens_pfx)

        orig_train = train_subs[:]
        # only filter train subjects
        train_subs = [s for s in train_subs if has_data(s)]
        dropped = set(orig_train) - set(train_subs)

        if dropped:
            print(f"[WARN] dropping train subjects missing {modality} data: {dropped}")

    # 3) build datasets
    ds_seed_train = None if synchronized else seed
    ds_seed_eval  = seed
    train_ds = FusionDataset(
        pose_dict, sensor_dict,
        subject_label_map, pose_label_map, sensor_label_map,
        train_subs, synchronized=synchronized, seed=ds_seed_train,
        pad_skel=pad_skel, pad_sens=pad_sens
    )
    eval_ds  = FusionDataset(
        pose_dict, sensor_dict,
        subject_label_map, pose_label_map, sensor_label_map,
        eval_subs,  synchronized=synchronized, seed=ds_seed_eval,
        pad_skel=pad_skel, pad_sens=pad_sens
    )

    if modality == "multimodal" and not synchronized:
        # get the current key‐lists
        pose_keys = train_ds.pose_ds.keys
        sens_keys = train_ds.sens_ds.keys
        n_pose, n_sens = len(pose_keys), len(sens_keys)

        if n_pose != n_sens:
            rng = random.Random(seed)
            if n_pose < n_sens:
                extra = rng.choices(pose_keys, k=(n_sens - n_pose))
                train_ds.pose_ds.keys = pose_keys + extra
            else:
                extra = rng.choices(sens_keys, k=(n_pose - n_sens))
                train_ds.sens_ds.keys = sens_keys + extra
                
    # 4) Class-balanced async eval for both walk & turn ───────────────────────
    if (modality=="multimodal") and not synchronized:
        # 1) Decide how to extract the grouping key per dataset
        if dataset == "walk":
            # two‐token prefix: "SUB01_on", "SUB02_off", etc.
            def subj_key(k): return "_".join(k.split("_")[:2])
        else:  # "turn"
            # one‐token prefix: "SUB01", "SUB02", etc.
            def subj_key(k): return k.split("_")[0]

        # 2) build subject → list of pose‐keys and sens‐keys
        pose_map = defaultdict(list)
        for k in eval_ds.pose_ds.keys:
            pose_map[subj_key(k)].append(k)

        sens_map = defaultdict(list)
        for k in eval_ds.sens_ds.keys:
            sens_map[subj_key(k)].append(k)

        # 3) compute target per-subject count = max across both modalities
        max_pose = max(len(pose_map[s]) for s in eval_subs)
        max_sens = max(len(sens_map[s]) for s in eval_subs)
        target  = max(max_pose, max_sens)

        # 4) oversample each subject up to `target`
        balanced_pose, balanced_sens = [], []
        for s in eval_subs:
            grp_p = pose_map.get(s, [])
            grp_s = sens_map.get(s, [])
            if not grp_p or not grp_s:
                raise ValueError(f"Subject {s} lacks data for one modality")
            for _ in range(target):
                balanced_pose.append(random.choice(grp_p))
                balanced_sens.append(random.choice(grp_s))

        random.shuffle(balanced_pose)
        random.shuffle(balanced_sens)

        # 5) replace the eval keys
        eval_ds.pose_ds.keys = balanced_pose
        eval_ds.sens_ds.keys = balanced_sens


    # 5) seed DataLoader workers
    g = torch.Generator()
    g.manual_seed(seed)

    # 6) build loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
        generator=g
    )
    eval_loader  = DataLoader(
        eval_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, eval_loader
