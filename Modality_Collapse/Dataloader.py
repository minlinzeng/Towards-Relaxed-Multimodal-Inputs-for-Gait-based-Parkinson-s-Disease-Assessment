import random
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

def pad_or_trim(seq: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad (at end) or trim a sequence to exactly `length` frames."""
    L = seq.shape[0]
    if L >= length:
        return seq[:length]
    pad = np.full((length - L, *seq.shape[1:]), pad_value, dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)

def center_poses(pose_dict: dict) -> dict:
    """Pelvis-center every pose."""
    ROOT = 0
    return {k: v - v[:, [ROOT], :] for k, v in pose_dict.items()}

def normalize_poses(pose_dict: dict) -> dict:
    """Min-max normalize each video into [0,1]."""
    out = {}
    for k, v in pose_dict.items():
        mn = v.min(axis=(0,1))
        mx = v.max(axis=(0,1))
        out[k] = (v - mn) / (mx - mn + 1e-6)
    return out

def seed_worker(worker_id):
    """Reproducible shuffling across workers."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

class SyncFusionDataset(Dataset):
    """
    Always returns *paired* (skeleton, sensor) and a single label.
    Single-modality models just ignore the unused branch.
    """
    def __init__(self, pose_dict, sensor_dict, label_map, subjects, pad_skel, pad_sens):
        # keep only keys whose prefix (SUBID) is in `subjects`
        def keep(k): return any(k.startswith(s) for s in subjects)
        skeys = {k for k in pose_dict   if keep(k)}
        gkeys = {k for k in sensor_dict if keep(k)}
        common = sorted(skeys & gkeys)   # require exact match in both
        self.pairs = common
        self.pose   = pose_dict
        self.sensor = sensor_dict
        self.labels = label_map
        self.ps, self.gs = pad_skel, pad_sens

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        k = self.pairs[i]
        xs = pad_or_trim(self.pose[k],   self.ps)
        xt = pad_or_trim(self.sensor[k], self.gs)
        sub = k.split("_")[0]
        y   = self.labels[sub]
        return {
            "skeleton":       torch.from_numpy(xs).float(),
            "sensor":         torch.from_numpy(xt).float(),
            "label_skeleton": torch.tensor(y, dtype=torch.long),
            "label_sensor":   torch.tensor(y, dtype=torch.long),
        }

def oversample_to_equal(pairs, label_map):
    """
    Given list of keys `pairs` and label_map: subject → class,
    return a new list where each class appears exactly `max_count` times.
    """
    # 1) group by class
    cls2keys = {}
    for k in pairs:
        cls = label_map[k.split("_")[0]]
        cls2keys.setdefault(cls, []).append(k)

    # 2) find maximum class size
    max_count = max(len(v) for v in cls2keys.values())

    # 3) oversample each class to max_count
    balanced = []
    for cls, keys in cls2keys.items():
        # if you want *exact* reproducibility, you could fix a seed here
        balanced += random.choices(keys, k=max_count)

    random.shuffle(balanced)
    return balanced

def create_balanced_fusion_loaders(reader, train_subjects, eval_subjects, batch_size, pad_skel, pad_sens, seed=0, num_workers=4):
    """
    Sync-only TURN loader.  Both single- and multi-modal models
    pull from exactly the same train / eval pairs.
    """
    # 1) preprocess skeletons
    pose   = center_poses(reader.pose_dict)
    pose   = normalize_poses(pose)
    sensor = reader.sensor_dict

    # 2) subject → label
    label_map = {
        s: (v[0] if isinstance(v, (list,tuple)) else int(v))
        for s, v in reader.labels_dict.items()
        if s not in ("SUB10","SUB30","SUB22")
    }

    # 3) datasets
    train_ds = SyncFusionDataset(pose, sensor, label_map, train_subjects, pad_skel, pad_sens)
    eval_ds  = SyncFusionDataset(pose, sensor, label_map, eval_subjects, pad_skel, pad_sens)

    # 4) build samplers
    train_ds.pairs = oversample_to_equal(train_ds.pairs, label_map)
    eval_ds.pairs  = oversample_to_equal(eval_ds.pairs,  label_map)

    # 5) reproducible generators
    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,         
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    eval_loader = DataLoader(
        eval_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        worker_init_fn=seed_worker, 
        generator=g
    )
    return train_loader, eval_loader
