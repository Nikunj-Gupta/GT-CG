#!/bin/bash

envs=(3m 8m 2s3z 3s5z 1c3s5z)
USE_CUDA="${USE_CUDA:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-gtcg-discovery}"

for e in "${envs[@]}"
do
   for i in {0..9}
   do
      python src/main.py --config=$1 --env-config=sc2 with env_args.map_name=$e seed=$i use_cuda="${USE_CUDA}" wandb_project="${WANDB_PROJECT}" &
      echo "Running with $1 and $e for seed=$i"
      sleep 2s
   done
done
