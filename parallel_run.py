import os
import copy
import argparse
import numpy as np
import multiprocessing as mp
from math import ceil
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from multitask_train import train_one_fold, set_random_seed, PDReader, pdfeReader, PATHS

LOG_DIR = "experiment_logs/d2/42hz/parallel_workers"
os.makedirs(LOG_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 41, 42, 43]
WEIGHTING_SCHEMES = [
    'none_reweighting',    # no weighted loss, no adaptive
    'weighted_loss_only',  # weighted loss only
    'adaptive_w_only',     # adaptive weighting only
    'wl_aw',               # both weighted loss & adaptive
]

def cross_validate_instance(reader, args):
    """Run full K‑fold CV for multitask, reusing a preloaded reader."""
    set_random_seed(args.seed)
    results = [train_one_fold(fold, reader, args) for fold in range(1, args.fold + 1)]
    avg_skel, avg_sensor = np.mean(results, axis=0)

    print(f"{args.fold}-Fold Results for seed={args.seed}, scheme="
          f"{'wl' if args.weighted_loss else ''}"
          f"{'aw' if args.use_adaptive_weighting else ''}:")
    for i, (sk, se) in enumerate(results, 1):
        print(f"  Fold {i}: skel={sk:.2f}%  sensor={se:.2f}%")
    print(f"  Average:   skel={avg_skel:.2f}%  sensor={avg_sensor:.2f}%  "
          f"combined={(avg_skel+avg_sensor)/2.0:.2f}%\n")

def train_single_instance(reader, args):
    """Run K‑fold training for single modality, reusing a preloaded reader."""
    set_random_seed(args.seed)
    modes = (['sensor','skeleton'] if args.modality=='both'
             else [args.modality])
    for mode in modes:
        args.modality = mode
        fold_accs = []
        for fold in range(1, args.fold + 1):
            acc = train_one_fold(fold, reader, args)
            fold_accs.append(acc if isinstance(acc, (int,float)) else acc[0])
        avg = np.mean(fold_accs)
        print(f"[{mode.upper()}] seed={args.seed}, scheme="
              f"{'wl' if args.weighted_loss else ''}"
              f"{'aw' if args.use_adaptive_weighting else ''}:")
        print("  Folds:", ["{:.2f}%".format(a) for a in fold_accs])
        print(f"  Average Best Eval Acc: {avg:.2f}%\n")

def worker_main(job_group, args_template):
    """
    Each worker:
      1) deep‑copies CLI args_template
      2) loads dataset (reader) once
      3) iterates over its assigned (seed,scheme) jobs
    """
    # 1) prepare args
    base_args = copy.deepcopy(args_template)
    if base_args.modality != "multimodal":
        base_args.use_adaptive_weighting = False

    # 2) load reader once
    if base_args.modality == "multimodal":
        ReaderCls, ctor_args = {
            "walk": (PDReader, (PATHS["walk"]["pose_path"], PATHS["walk"]["sensor_path"], PATHS["walk"]["label_path"])),
            "turn": (pdfeReader, (PATHS["turn"]["pose_path"], PATHS["turn"]["sensor_path"], PATHS["turn"]["label_path"]))
        }[base_args.dataset]
        reader = ReaderCls(*ctor_args)
    else: # single‑task still uses the pdfeReader for pose+sensor
        reader = pdfeReader(PATHS[base_args.dataset]["pose_path"], PATHS[base_args.dataset]["sensor_path"], PATHS[base_args.dataset]["label_path"])

    # 3) run each experiment in this group
    for seed, scheme in job_group:
        args = copy.deepcopy(base_args)
        args.seed = seed
        args.weighted_loss = scheme in ('weighted_loss_only','wl_aw')
        args.use_adaptive_weighting = scheme in ('adaptive_w_only', 'wl_aw')

        log_file = os.path.join(LOG_DIR, f"s{seed}_{scheme}.log")
        with open(log_file, "a") as lf, redirect_stdout(lf), redirect_stderr(lf):
            def log(msg):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lf.write(f"[{ts}] {msg}\n")
                lf.flush()

            log(f"START seed={seed}, scheme={scheme}")
            try:
                if args.modality == "multimodal": cross_validate_instance(reader, args)
                else: train_single_instance(reader, args)
                log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] DONE\n")
            except Exception as e:
                log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ERROR: {e!r}\n")

if __name__ == "__main__":
    # use spawn so CUDA & DataLoader workers are safe
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  type=str, default="turn", help="walk or turn")
    p.add_argument("--modality", type=str, default="multimodal", choices=["multimodal","skeleton","sensor","both"])
    p.add_argument("--fold",     type=int, default=7, help="number of CV folds")
    args = p.parse_args()

    if args.modality == "multimodal": schemes = WEIGHTING_SCHEMES
    else: schemes = WEIGHTING_SCHEMES[:2]  # no adaptive for single task
    jobs = [(s, w) for s in SEEDS for w in schemes]

    N = min(mp.cpu_count(), len(jobs))
    chunk = ceil(len(jobs) / N)
    job_groups = [jobs[i:i+chunk] for i in range(0, len(jobs), chunk)]

    # spawn one non‑daemon process per group
    processes = []
    for group in job_groups:
        p = mp.Process(target=worker_main, args=(group, args))
        p.daemon = False
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
