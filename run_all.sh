#!/usr/bin/env bash
set -e

# where to write logs
LOGDIR=logs/sync_train_sync_test
mkdir -p $LOGDIR

# list your fusion variants and seeds
fusion_types=(early late share_latent cheap_xattn)
seeds=(0 1 2 3 4 40 41 42 43 44)

# available GPU IDs
gpus=(0 1 )
ngpu=${#gpus[@]}

for ft in "${fusion_types[@]}"; do
  for idx in "${!seeds[@]}"; do
    seed=${seeds[$idx]}
    gpu=${gpus[$(( idx % ngpu ))]}

    echo "Launching fusion_type=$ft seed=$seed on GPU $gpu…"
    CUDA_VISIBLE_DEVICES=$gpu nohup python baseline/fusion_train.py \
      --fusion_type $ft \
      --seed $seed \
      --synchronized_loading \
    > $LOGDIR/log_sync_balanced_${ft}_s${seed}.out 2>&1 &

    # slight pause to avoid race on stdout
    sleep 0.5
  done
done

echo "All jobs dispatched."
