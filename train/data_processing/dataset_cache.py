from __future__ import annotations

import pickle
import sys
import argparse
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "train"
for path in (PROJECT_ROOT, TRAIN_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from const.path import get_pd_paths
try:
    from configs import normalize_dataset_name, raw_reader_dataset_name, WEARGAIT_PREPROC_DIR
except ImportError:
    from train.configs import normalize_dataset_name, raw_reader_dataset_name, WEARGAIT_PREPROC_DIR


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
WEARGAIT_CACHE_DIR = PROJECT_ROOT / WEARGAIT_PREPROC_DIR


def reader_cache_path(dataset: str, cache_dir: str | Path | None = None) -> Path:
    dataset = normalize_dataset_name(dataset)
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return cache_root / f"{dataset}_reader.pkl"


def legacy_reader_cache_path(dataset: str, cache_dir: str | Path | None = None) -> Path:
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return cache_root / f"{raw_reader_dataset_name(dataset)}_reader.pkl"


def build_reader(dataset: str) -> Any:
    dataset = normalize_dataset_name(dataset)
    raw_dataset = raw_reader_dataset_name(dataset)
    paths = get_pd_paths()
    if dataset == "fbg":
        from data_processing.preprocess_fbg import PDReader

        p = paths[raw_dataset]
        return PDReader(
            joints_path=p["pose_path"],
            sensor_path=p["sensor_path"],
            labels_path=p["label_path"],
        )
    if dataset == "fog":
        from data_processing.preprocess_fog import pdfeReader

        p = paths[raw_dataset]
        return pdfeReader(
            pose_path=p["pose_path"],
            sensor_path=p["sensor_path"],
            label_path=p["label_path"],
            lifted_path=p["lifted_path"],
        )
    raise ValueError(f"Unknown cached reader dataset: {dataset}")


def _try_load_pickle(path: Path, dataset: str, cache_label: str) -> Any | None:
    print(f"[CACHE] Loading {dataset} reader from {cache_label} {path}")
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, ImportError, AttributeError, pickle.UnpicklingError, EOFError) as exc:
        print(f"[CACHE] Could not load {dataset} reader cache {path}: {exc}. Rebuilding cache.")
        return None


def load_reader(dataset: str, *, rebuild: bool = False, cache_dir: str | Path | None = None) -> Any:
    dataset = normalize_dataset_name(dataset)
    path = reader_cache_path(dataset, cache_dir)
    if path.exists() and not rebuild:
        reader = _try_load_pickle(path, dataset, "cache")
        if reader is not None:
            return reader

    legacy_path = legacy_reader_cache_path(dataset, cache_dir)
    if legacy_path.exists() and not rebuild:
        reader = _try_load_pickle(legacy_path, dataset, "legacy cache")
        if reader is not None:
            return reader

    print(f"[CACHE] Building {dataset} reader and saving to {path}")
    reader = build_reader(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(reader, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    return reader


def summarize_reader(dataset: str, reader: Any) -> dict[str, int]:
    dataset = normalize_dataset_name(dataset)
    if dataset == "fbg":
        return {
            "pose_entries": len(reader.pose_dict),
            "sensor_entries": len(reader.sensor_dict),
            "pose_labels": len(reader.pose_label_dict),
            "sensor_labels": len(reader.sensor_label_dict),
        }
    if dataset == "fog":
        return {
            "pose_entries": len(reader.pose_dict),
            "sensor_entries": len(reader.sensor_dict),
            "subject_labels": len(reader.labels_dict),
            "sensor_length": int(reader.sensor_length),
        }
    raise ValueError(f"Unknown cached reader dataset: {dataset}")


def count_weargait_pickles(data_dir: str | Path | None = None) -> int:
    root = Path(data_dir) if data_dir is not None else WEARGAIT_CACHE_DIR
    return len(list(root.glob("*.pkl"))) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser("Generate reusable dataset pickle caches")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["fbg", "fog", "weargait", "all"],
        default=["all"],
        help="Datasets to cache. 'all' means FBG, FoG, and WearGait verification.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild existing FBG/FoG reader caches.")
    args = parser.parse_args()

    requested = ["fbg", "fog", "weargait"] if "all" in args.datasets else args.datasets

    for dataset in requested:
        if dataset == "weargait":
            count = count_weargait_pickles()
            if count == 0:
                raise FileNotFoundError(
                    "No WearGait .pkl files found. Run train/data_processing/preprocess_weargait.py first."
                )
            print(f"[CACHE] WearGait already has {count} per-subject .pkl files.")
            continue

        reader = load_reader(dataset, rebuild=args.rebuild)
        summary = summarize_reader(dataset, reader)
        print(f"[CACHE] {dataset}: {summary}")


if __name__ == "__main__":
    main()
