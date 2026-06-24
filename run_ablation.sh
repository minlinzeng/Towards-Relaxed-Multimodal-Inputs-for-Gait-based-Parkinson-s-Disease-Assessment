#!/usr/bin/env bash
set -euo pipefail

# Canonical ablation launcher.
#
# Log layout:
#   results/ablation/logs/{sync|async}/{fbg|fog|weargait}/{cr|no_cr}/{baseline}/seed{seed}.out
#
# Examples:
#   RUN_MODE=async DATASET=weargait ./run_ablation.sh
#   DETACH=1 RUN_MODE=async DATASET=weargait BASELINES_STR="early_fusion" ./run_ablation.sh
#   RUN_MODE=async DATASET=fog BASELINES_STR="ours focal taca" ./run_ablation.sh
#   RUN_MODE=sync DATASET=fog BASELINES_STR="early_fusion late_fusion" ./run_ablation.sh

LOG_ROOT_WAS_SET=1
if [[ -z "${LOG_ROOT+x}" ]]; then
  LOG_ROOT_WAS_SET=0
  LOG_ROOT=results/ablation/logs
fi
RUN_MODE=${RUN_MODE:-async}
DATASET=${DATASET:-weargait}
OURS_WM=${OURS_WM:-${WM:-gcl}}
OURS_PARAM_PRESET=${OURS_PARAM_PRESET:-}
OURS_PARAM_PRESETS_STR=${OURS_PARAM_PRESETS_STR:-}
OURS_LOG_NAME=${OURS_LOG_NAME:-ours}
BASELINE_WM=${BASELINE_WM:-class_wt}
OURS_REBALANCE_TAG=${OURS_REBALANCE_TAG:-}
BASELINE_REBALANCE_TAG=${BASELINE_REBALANCE_TAG:-}
BASELINES_STR=${BASELINES_STR:-"ours early_fusion late_fusion shared_latent cheap_xattn deepav_lite focal taca"}
SEEDS_STR=${SEEDS_STR:-"2 3 4 42 43 44"}
GPUS_STR=${GPUS_STR:-"0"}
JOBS_PER_GPU=${JOBS_PER_GPU:-30}
SLEEP_SEC=${SLEEP_SEC:-0.2}
DETACH=${DETACH:-0}
WAIT_FOR_JOBS=${WAIT_FOR_JOBS:-0}
DISPATCH_LOG=${DISPATCH_LOG:-}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/yongjie/anaconda3/envs/minlin310/bin/python" ]]; then
    PYTHON_BIN=/home/yongjie/anaconda3/envs/minlin310/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi
MISSING_RATIO_EVAL=${MISSING_RATIO_EVAL:-0}
MISSING_RATIOS=${MISSING_RATIOS:-"0.2,0.4,0.6,0.8"}
MISSING_RATIO_REPEATS=${MISSING_RATIO_REPEATS:-5}

if [[ "$RUN_MODE" != "sync" && "$RUN_MODE" != "async" ]]; then
  echo "RUN_MODE must be 'sync' or 'async'." >&2
  exit 1
fi
if [[ "$DATASET" != "fbg" && "$DATASET" != "fog" && "$DATASET" != "weargait" ]]; then
  echo "DATASET must be 'fbg', 'fog', or 'weargait'." >&2
  exit 1
fi
if ! [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS_PER_GPU must be a positive integer." >&2
  exit 1
fi
if [[ "$RUN_MODE" == "sync" && "$DATASET" == "fbg" ]]; then
  echo "FBG synchronous evaluation is intentionally disabled." >&2
  echo "Reason: TRIP_TKDD.pdf omits FBG sync results because inconsistent temporal alignment across sensor trials prevents fair synchronized evaluation." >&2
  echo "Use RUN_MODE=async DATASET=fbg, or RUN_MODE=sync with DATASET=fog/weargait." >&2
  exit 1
fi
if [[ "$LOG_ROOT_WAS_SET" == "0" && "$RUN_MODE" == "sync" && "$MISSING_RATIO_EVAL" == "1" ]]; then
  LOG_ROOT=results/ablation/logs_missing_ratio
fi

infer_rebalance_tag() {
  local wm=$1 preset=${2:-}
  if [[ "$preset" == *"_ce_"* || "$preset" == *"_ce" ]]; then
    echo "no_cr"
  elif [[ "$wm" == "ce" ]]; then
    echo "no_cr"
  else
    echo "cr"
  fi
}

baseline_rebalance_tag() {
  local baseline=$1
  if [[ "$baseline" == "ours" ]]; then
    if [[ -n "$OURS_REBALANCE_TAG" ]]; then
      echo "$OURS_REBALANCE_TAG"
    else
      infer_rebalance_tag "$OURS_WM" "$OURS_PARAM_PRESET"
    fi
  else
    if [[ -n "$BASELINE_REBALANCE_TAG" ]]; then
      echo "$BASELINE_REBALANCE_TAG"
    else
      infer_rebalance_tag "$BASELINE_WM" ""
    fi
  fi
}

if [[ "$DETACH" == "1" && "${ABLATION_LAUNCHER_CHILD:-0}" != "1" ]]; then
  dispatch_dir="$LOG_ROOT/$RUN_MODE/$DATASET"
  mkdir -p "$dispatch_dir"
  dispatch_log=${DISPATCH_LOG:-"$dispatch_dir/dispatch.out"}

  echo "Starting detached ablation launcher -> $dispatch_log"
  ABLATION_LAUNCHER_CHILD=1 nohup bash "$0" > "$dispatch_log" 2>&1 &
  echo "Launcher PID: $!"
  exit 0
fi

read -r -a baselines <<< "$BASELINES_STR"
read -r -a seeds <<< "$SEEDS_STR"
read -r -a gpus <<< "$GPUS_STR"
ngpu=${#gpus[@]}
if [[ "$ngpu" -lt 1 ]]; then
  echo "GPUS_STR must contain at least one GPU id." >&2
  exit 1
fi

fbg_fog_sync_args=()
baseline_sync_args=()
weargait_mode_args=()
missing_ratio_args=()
active_pids=()
active_gpus=()
if [[ "$RUN_MODE" == "sync" ]]; then
  fbg_fog_sync_args=(--synchronized_loading)
  baseline_sync_args=(--synced)
else
  weargait_mode_args=(--async_loading)
fi
if [[ "$RUN_MODE" == "sync" && "$MISSING_RATIO_EVAL" == "1" ]]; then
  missing_ratio_args=(--eval_missing_ratios --missing_ratios "$MISSING_RATIOS" --missing_ratio_repeats "$MISSING_RATIO_REPEATS")
fi

reap_finished_jobs() {
  local kept_pids=()
  local kept_gpus=()
  local idx pid
  for idx in "${!active_pids[@]}"; do
    pid=${active_pids[$idx]}
    if kill -0 "$pid" 2>/dev/null; then
      kept_pids+=("$pid")
      kept_gpus+=("${active_gpus[$idx]}")
    else
      wait "$pid" 2>/dev/null || true
    fi
  done
  active_pids=("${kept_pids[@]}")
  active_gpus=("${kept_gpus[@]}")
}

active_count_for_gpu() {
  local gpu=$1
  local count=0
  local idx
  for idx in "${!active_pids[@]}"; do
    if [[ "${active_gpus[$idx]}" == "$gpu" ]]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

wait_for_gpu_slot() {
  local gpu=$1
  local active_count
  while true; do
    reap_finished_jobs
    active_count=$(active_count_for_gpu "$gpu")
    if [[ "$active_count" -lt "$JOBS_PER_GPU" ]]; then
      return
    fi
    echo "GPU $gpu has $active_count active job(s); waiting for a free slot..."
    sleep 30
  done
}

fusion_type_for() {
  case "$1" in
    early_fusion) echo "early" ;;
    late_fusion) echo "late" ;;
    shared_latent) echo "share_latent" ;;
    cheap_xattn) echo "cheap_xattn" ;;
    *) return 1 ;;
  esac
}

launch_fbg_fog() {
  local baseline=$1 seed=$2 gpu=$3 logfile=$4 preset=${5:-}
  LAST_LAUNCHED_PID=

  case "$baseline" in
    ours)
      local ours_preset_args=()
      if [[ -n "$preset" ]]; then
        ours_preset_args=(--param_preset "$preset")
      elif [[ -n "$OURS_PARAM_PRESET" ]]; then
        ours_preset_args=(--param_preset "$OURS_PARAM_PRESET")
      fi
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u -m train.train --mode fbg_fog \
        --dataset "$DATASET" \
        --modality multimodal \
        --wm "$OURS_WM" \
        --seed "$seed" \
        "${ours_preset_args[@]}" \
        "${fbg_fog_sync_args[@]}" \
        "${missing_ratio_args[@]}" \
        > "$logfile" 2>&1 &
      LAST_LAUNCHED_PID=$!
      ;;
    early_fusion|late_fusion|shared_latent|cheap_xattn)
      local fusion_type
      fusion_type=$(fusion_type_for "$baseline")
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u -m train.train --mode fusion \
        --dataset "$DATASET" \
        --fusion_type "$fusion_type" \
        --wm "$BASELINE_WM" \
        --seed "$seed" \
        "${fbg_fog_sync_args[@]}" \
        "${missing_ratio_args[@]}" \
        > "$logfile" 2>&1 &
      LAST_LAUNCHED_PID=$!
      ;;
    deepav_lite)
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u -m train.train --mode deepav \
        --dataset "$DATASET" \
        --wm "$BASELINE_WM" \
        --seed "$seed" \
        "${baseline_sync_args[@]}" \
        "${missing_ratio_args[@]}" \
        > "$logfile" 2>&1 &
      LAST_LAUNCHED_PID=$!
      ;;
    focal)
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u -m train.train --mode focal \
        --dataset "$DATASET" \
        --wm "$BASELINE_WM" \
        --seed "$seed" \
        "${baseline_sync_args[@]}" \
        "${missing_ratio_args[@]}" \
        > "$logfile" 2>&1 &
      LAST_LAUNCHED_PID=$!
      ;;
    taca)
      CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u -m train.train --mode taca \
        --dataset "$DATASET" \
        --wm "$BASELINE_WM" \
        --seed "$seed" \
        "${baseline_sync_args[@]}" \
        "${missing_ratio_args[@]}" \
        > "$logfile" 2>&1 &
      LAST_LAUNCHED_PID=$!
      ;;
    *)
      echo "Unknown baseline for $DATASET: $baseline" >&2
      exit 1
      ;;
  esac
}

launch_weargait() {
  local baseline=$1 seed=$2 gpu=$3 logfile=$4
  local baseline_args=()
  local run_wm=$OURS_WM
  LAST_LAUNCHED_PID=
  if [[ "$baseline" != "ours" ]]; then
    baseline_args=(--baseline "$baseline")
    run_wm=$BASELINE_WM
  fi

  CUDA_VISIBLE_DEVICES=$gpu nohup "$PYTHON_BIN" -u train/weargait_train.py \
    --seed "$seed" \
    --wm "$run_wm" \
    "${weargait_mode_args[@]}" \
    "${missing_ratio_args[@]}" \
    "${baseline_args[@]}" \
    > "$logfile" 2>&1 &
  LAST_LAUNCHED_PID=$!
}

for baseline in "${baselines[@]}"; do
  preset_labels=("")
  if [[ "$baseline" == "ours" && -n "$OURS_PARAM_PRESETS_STR" ]]; then
    read -r -a preset_labels <<< "$OURS_PARAM_PRESETS_STR"
  fi
  for current_ours_preset in "${preset_labels[@]}"; do
  for idx in "${!seeds[@]}"; do
    seed=${seeds[$idx]}
    gpu=${gpus[$((idx % ngpu))]}
    log_label="$baseline"
    if [[ "$baseline" == "ours" ]]; then
      if [[ -n "$current_ours_preset" ]]; then
        log_label="$current_ours_preset"
      else
        log_label="$OURS_LOG_NAME"
      fi
    fi
    rebalance_tag=$(baseline_rebalance_tag "$baseline")
    logdir="$LOG_ROOT/$RUN_MODE/$DATASET/$rebalance_tag/$log_label"
    logfile="$logdir/seed${seed}.out"
    mkdir -p "$logdir"

    wait_for_gpu_slot "$gpu"
    echo "Launching mode=$RUN_MODE dataset=$DATASET baseline=$baseline seed=$seed on GPU $gpu -> $logfile"

    if [[ "$DATASET" == "weargait" ]]; then
      launch_weargait "$baseline" "$seed" "$gpu" "$logfile"
    else
      launch_fbg_fog "$baseline" "$seed" "$gpu" "$logfile" "$current_ours_preset"
    fi

    active_pids+=("$LAST_LAUNCHED_PID")
    active_gpus+=("$gpu")
    echo "Launched PID=$LAST_LAUNCHED_PID (active on GPU $gpu: $(active_count_for_gpu "$gpu")/$JOBS_PER_GPU)"

    sleep "$SLEEP_SEC"
  done
  done
done

echo "All ablation jobs dispatched."

if [[ "$WAIT_FOR_JOBS" == "1" ]]; then
  echo "WAIT_FOR_JOBS=1; waiting for ${#active_pids[@]} active job(s) to finish..."
  for pid in "${active_pids[@]}"; do
    wait "$pid"
  done
  echo "All ablation jobs finished."
fi


# DETACH=1 RUN_MODE=async DATASET=weargait BASELINES_STR="early_fusion" SEEDS_STR="2 3 4 42 43 44" ./run_ablation.sh
# DETACH=1 RUN_MODE=async DATASET=fog BASELINES_STR="ours" OURS_PARAM_PRESET=fog_best_param SEEDS_STR="2 3 4 42 43 44" GPUS_STR="0" ./run_ablation.sh
# DETACH=1 RUN_MODE=sync DATASET=weargait MISSING_RATIO_EVAL=1 BASELINES_STR="ours early_fusion cheap_xattn" ./run_ablation.sh
# DETACH=1 RUN_MODE=sync DATASET=fog MISSING_RATIO_EVAL=1 BASELINES_STR="ours early_fusion cheap_xattn" OURS_PARAM_PRESET=fog_sync_best_param ./run_ablation.sh
