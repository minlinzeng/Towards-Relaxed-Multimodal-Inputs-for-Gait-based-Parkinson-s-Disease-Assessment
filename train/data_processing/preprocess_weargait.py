from pathlib import Path
import json, re
import numpy as np
import pandas as pd

# ========================= Config =========================
GRAV = 9.81
IMU_SITES = [
    "L_Ankle","R_Ankle",
    "L_DorsalFoot","R_DorsalFoot",
    "L_MidLatThigh","R_MidLatThigh",
    "L_LatShank","R_LatShank",
]
CSV_PATTERN = "*_SelfPace_matTURN.csv"   # adjust if needed
hc_path = "data/WearGait/HC"
pd_path = "data/WearGait/PD"
hc_demo_csv = "data/WearGait/HC/hc_demographic.csv"
pd_demo_csv = "data/WearGait/PD/pd_demographic.csv"
output_dir = "data/WearGait/WearGait_preproc_SPmT_30Hz"

# ==================== Demographics I/O ====================
def read_demographics_with_header_fix(path: str) -> pd.DataFrame:
    df0 = pd.read_csv(path, header=None, dtype=str)
    hdr_idx = 1
    header = df0.iloc[hdr_idx].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    df = df0.iloc[hdr_idx+1:].reset_index(drop=True).copy()
    df.columns = header
    return df

def extract_subject_weights(demo_df: pd.DataFrame) -> pd.DataFrame:
    id_col = next(c for c in demo_df.columns if re.search(r"(subject\s*id|participant)", c, re.I))
    wt_col = next(c for c in demo_df.columns if re.search(r"weight", c, re.I))
    out = demo_df[[id_col, wt_col]].rename(columns={id_col:"subject_id", wt_col:"weight_kg"}).copy()
    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    out["weight_kg"] = pd.to_numeric(out["weight_kg"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0], errors="coerce")
    return out.dropna(subset=["subject_id","weight_kg"]).reset_index(drop=True)

def build_weight_map(hc_demo_csv: str, pd_demo_csv: str) -> dict:
    weight_map = {}
    for p in [hc_demo_csv, pd_demo_csv]:
        if not p: 
            continue
        df = read_demographics_with_header_fix(p)
        w = extract_subject_weights(df)
        weight_map.update({r.subject_id.lower(): float(r.weight_kg) for _, r in w.iterrows()})
    return weight_map

# ==================== File discovery ======================
def find_subject_files(root_dir: str, pattern: str = CSV_PATTERN) -> dict:
    root = Path(root_dir)
    return {p.stem.split("_",1)[0].lower(): p for p in root.glob(pattern)}

# ===================== IMU train stats ====================
def list_imu_freeacc_cols(cols) -> list:
    out = []
    for s in IMU_SITES:
        for ax in ["E","N","U"]:
            c = f"{s}_FreeAcc_{ax}"
            if c in cols:
                out.append(c)
    for side in ["Linsole","Rinsole"]:
        for ax in ["X","Y","Z"]:
            c = f"{side}:Acc_{ax}"
            if c in cols:
                out.append(c)
    return out

def fit_train_stats(train_csv_paths: list) -> dict:
    if not train_csv_paths:
        raise ValueError("Empty training list for IMU normalization.")
    sample_cols = pd.read_csv(train_csv_paths[0], nrows=0).columns
    channels = list_imu_freeacc_cols(sample_cols)

    sums = {c:0.0 for c in channels}
    sumsqs = {c:0.0 for c in channels}
    counts = {c:0   for c in channels}

    for p in train_csv_paths:
        df = pd.read_csv(p)
        for c in channels:
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").to_numpy()
                m = np.isfinite(x)
                n = int(m.sum())
                if n:
                    x = x[m]
                    sums[c]   += float(x.sum())
                    sumsqs[c] += float(np.dot(x, x))
                    counts[c] += n

    stats = {}
    for c in channels:
        n = counts[c]
        if n > 0:
            mean = sums[c]/n
            var  = max((sumsqs[c]/n) - mean**2, 0.0)
            std  = max(np.sqrt(var), 1e-8)
        else:
            mean, std = 0.0, 1.0
        stats[c] = (mean, std)
    return stats

def apply_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    for c,(m,s) in stats.items():
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").to_numpy()
            out[c] = (x - m) / (s if s != 0 else 1.0)
    return out

# ==================== Downsampling ========================
def parse_time_seconds(s: pd.Series) -> np.ndarray:
    t = (s.astype(str)
           .str.strip()
           .str.replace(" sec", "", regex=False)
           .str.replace(",", ".", regex=False))
    return pd.to_numeric(t, errors="coerce").to_numpy(dtype=float)

def downsample_to_30hz(df: pd.DataFrame, time_col="Time", target_hz=30) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df
    t = parse_time_seconds(df[time_col])
    m = np.isfinite(t)
    if not m.any():
        return pd.DataFrame()
    bins = np.full(t.shape, -1, dtype=np.int64)
    bins[m] = np.floor(t[m] * target_hz).astype(np.int64)

    tmp = df.copy()
    tmp["_bin"] = bins
    out = tmp[tmp["_bin"] >= 0].groupby("_bin", sort=True, as_index=False).first()
    out[time_col] = (out["_bin"].to_numpy(dtype=float) + 0.5) / target_hz
    out = out.drop(columns=["_bin"]).reset_index(drop=True)
    return out

# ==================== Stream builders =====================
def build_walkway(df: pd.DataFrame, weight_kg: float) -> pd.DataFrame:
    keep = [c for c in ["Time","L Foot Pressure","R Foot Pressure"] if c in df.columns]
    if not keep: 
        return pd.DataFrame()
    out = df[keep].copy()

    denom = weight_kg * GRAV if weight_kg and weight_kg > 0 else np.nan
    for c in ["L Foot Pressure","R Foot Pressure"]:
        if c in out and denom:
            out[c + "_BW"] = pd.to_numeric(out[c], errors="coerce") / denom

    cols = ["Time"] + [c for c in ["L Foot Pressure_BW","R Foot Pressure_BW"] if c in out.columns]
    out = out[cols]
    return downsample_to_30hz(out)

def build_insole(df: pd.DataFrame, weight_kg: float, stats: dict | None) -> pd.DataFrame:
    keep = [c for c in [
        "Time","LTotalForce","RTotalForce",
        "LCoP_X","LCoP_Y","RCoP_X","RCoP_Y",
        "Linsole:Acc_X","Linsole:Acc_Y","Linsole:Acc_Z",
        "Rinsole:Acc_X","Rinsole:Acc_Y","Rinsole:Acc_Z",
    ] if c in df.columns]
    if not keep: 
        return pd.DataFrame()
    out = df[keep].copy()

    # BW normalize forces (always safe and fold-agnostic)
    if weight_kg and weight_kg > 0:
        denom = weight_kg * GRAV
        for c in ["LTotalForce","RTotalForce"]:
            if c in out:
                out[c+"_BW"] = pd.to_numeric(out[c], errors="coerce") / denom
        if {"LTotalForce","RTotalForce"}.issubset(out.columns):
            out["SumForce_BW"] = (pd.to_numeric(out["LTotalForce"], errors="coerce") +
                                  pd.to_numeric(out["RTotalForce"], errors="coerce")) / denom

    # z-score insole Acc only if stats provided  # <<< minimal change
    if stats is not None:
        for side in ["Linsole","Rinsole"]:
            for ax in ["X","Y","Z"]:
                col = f"{side}:Acc_{ax}"
                if col in out.columns and col in stats:
                    m, s = stats[col]
                    x = pd.to_numeric(out[col], errors="coerce").to_numpy()
                    out[col] = (x - m) / (s if s != 0 else 1.0)

    # pack tuples
    def _pack(prefix):
        cols = [f"{prefix}:Acc_X", f"{prefix}:Acc_Y", f"{prefix}:Acc_Z"]
        if all(c in out.columns for c in cols):
            out[f"{prefix}_Acc"] = list(map(tuple, out[cols].to_numpy()))
            out.drop(columns=cols, inplace=True)
    _pack("Linsole"); _pack("Rinsole")

    cols = ["Time","LTotalForce_BW","RTotalForce_BW","SumForce_BW",
            "LCoP_X","LCoP_Y","RCoP_X","RCoP_Y","Linsole_Acc","Rinsole_Acc"]
    out = out[[c for c in cols if c in out.columns]]
    return downsample_to_30hz(out)

def build_imu(df: pd.DataFrame, stats: dict | None) -> pd.DataFrame:
    keep = ["Time"]
    for s in IMU_SITES:
        keep += [c for c in [f"{s}_FreeAcc_E", f"{s}_FreeAcc_N", f"{s}_FreeAcc_U"] if c in df.columns]
    if len(keep) == 1:
        return pd.DataFrame()

    imu = df[[c for c in keep if c in df.columns]].copy()

    # z-score only if stats provided  # <<< minimal change
    if stats is not None:
        for s in IMU_SITES:
            for ax in ["E","N","U"]:
                c = f"{s}_FreeAcc_{ax}"
                if c in imu.columns and c in stats:
                    m, sdev = stats[c]
                    x = pd.to_numeric(imu[c], errors="coerce").to_numpy()
                    imu[c] = (x - m) / (sdev if sdev != 0 else 1.0)

    # pack per site
    for s in IMU_SITES:
        cols = [f"{s}_FreeAcc_E", f"{s}_FreeAcc_N", f"{s}_FreeAcc_U"]
        if all(c in imu.columns for c in cols):
            imu[f"{s}_FreeAcc"] = list(map(tuple, imu[cols].to_numpy()))
            imu.drop(columns=cols, inplace=True)

    return downsample_to_30hz(imu)

# ====================== Orchestrator ======================
def run_end_to_end(
    hc_csv_root: str,
    pd_csv_root: str,
    hc_demo_csv: str,
    pd_demo_csv: str,
    output_dir: str,
    train_subject_ids: list | None,   # None → skip fitting, save *_base.pkl
    pattern: str = CSV_PATTERN,
    segment_len_rows: int | None = None,   # e.g., 300 rows = 10 s @30 Hz
    segment_len_sec: float | None = None   # if given, overrides rows
):
    """
    Processes WearGait CSVs → PKLs and prints per-subject and total counts.
    If train_subject_ids is None:
      - No IMU z-score is fitted/applied.
      - Files saved as *_imu_base.pkl / *_insole_base.pkl.
    Otherwise:
      - Fit on those subjects and save normalized *_imu.pkl / *_insole.pkl.
    """
    HZ = 30  # matches downsample_to_30hz default
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # segment size (rows)
    if segment_len_sec is not None:
        seg_rows = int(max(1, np.floor(float(segment_len_sec) * HZ)))
    elif segment_len_rows is not None:
        seg_rows = int(max(1, segment_len_rows))
    else:
        seg_rows = None

    # weights
    weight_map = build_weight_map(hc_demo_csv, pd_demo_csv)

    # files
    files_hc = find_subject_files(hc_csv_root, pattern)
    files_pd = find_subject_files(pd_csv_root, pattern)
    all_files = {**files_hc, **files_pd}
    if not all_files:
        print("[warn] no CSV files found; check paths/pattern"); 
        return

    # fit stats if requested
    stats = None
    if train_subject_ids:
        train_paths = []
        for sid in train_subject_ids:
            p = all_files.get(str(sid).lower())
            if p is not None:
                train_paths.append(str(p))
        if not train_paths:
            raise ValueError("No training CSVs found. Check train_subject_ids or pattern.")
        stats = fit_train_stats(train_paths)

    # accumulators
    total_rows_w = total_rows_i = total_rows_m = total_rows_any = 0
    total_segs_w = total_segs_i = total_segs_m = total_segs_all = 0

    # process subjects
    for sid_lower, csv_path in all_files.items():
        df = pd.read_csv(csv_path)
        if "GeneralEvent" in df.columns:
            df = df[df["GeneralEvent"].str.lower() != "standing"].copy()
        wkg = weight_map.get(sid_lower, np.nan)

        walkway = build_walkway(df, wkg)
        insole  = build_insole(df, wkg, stats)
        imu     = build_imu(df, stats)

        # counts
        nw, ni, nm = len(walkway), len(insole), len(imu)
        n_any = max(nw, ni, nm)
        secs_any = n_any / HZ if n_any > 0 else 0.0

        if seg_rows is not None:
            sw = nw // seg_rows
            si = ni // seg_rows
            sm = nm // seg_rows
            sall = min(nw, ni, nm) // seg_rows
        else:
            sw = si = sm = sall = 0

        # print per-subject
        print(
            f"[{sid_lower}] rows_w={nw} rows_i={ni} rows_m={nm} "
            f"rows_any={n_any} secs_any={secs_any:.3f}"
            + (f" | seg_rows={seg_rows} segs_w={sw} segs_i={si} segs_m={sm} segs_all={sall}" if seg_rows else "")
        )

        # save PKLs
        if stats is None:
            walkway.to_pickle(outdir / f"{sid_lower}_walkway.pkl")
            insole.to_pickle (outdir / f"{sid_lower}_insole_base.pkl")
            imu.to_pickle     (outdir / f"{sid_lower}_imu_base.pkl")
        else:
            walkway.to_pickle(outdir / f"{sid_lower}_walkway.pkl")
            insole.to_pickle (outdir / f"{sid_lower}_insole.pkl")
            imu.to_pickle     (outdir / f"{sid_lower}_imu.pkl")

        # accumulate totals
        total_rows_w   += nw
        total_rows_i   += ni
        total_rows_m   += nm
        total_rows_any += n_any
        total_segs_w   += sw
        total_segs_i   += si
        total_segs_m   += sm
        total_segs_all += sall

    # persist stats if fitted
    if stats is not None:
        with open(outdir / "imu_freeacc_stats.json", "w") as f:
            json.dump(stats, f)

    # totals
    print(
        f"[TOTAL] rows_w={total_rows_w} rows_i={total_rows_i} rows_m={total_rows_m} "
        f"rows_any={total_rows_any} secs_any={total_rows_any / HZ:.3f}"
        + (f" | seg_rows={seg_rows} segs_w={total_segs_w} segs_i={total_segs_i} segs_m={total_segs_m} segs_all={total_segs_all}" if seg_rows else "")
    )

def main() -> None:
    # Fold-agnostic preprocessing (recommended before CV).
    run_end_to_end(hc_path, pd_path, hc_demo_csv, pd_demo_csv, output_dir, train_subject_ids=None)


if __name__ == "__main__":
    main()

# Per-fold, pass the training IDs for that fold to get fold-specific normalized PKLs:
# run_end_to_end(hc_path, pd_path, hc_demo_csv, pd_demo_csv, f"{output_dir}/fold_0", train_subject_ids=fold0_train_ids)
