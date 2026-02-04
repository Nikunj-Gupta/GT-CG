#!/usr/bin/env bash
#SBATCH --account=prasanna_1363
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-59%32

conda activate gtcg

ENVS=("aloha" "disperse" "gather" "hallway" "pursuit" "sensor")
ALGOS=("gtcg" "gpscg")
SEEDS=("0" "1" "2" "3" "4")

NUM_ENVS=${#ENVS[@]}
NUM_ALGOS=${#ALGOS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_ENVS * NUM_ALGOS * NUM_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid TASK_ID ${TASK_ID} (TOTAL=${TOTAL})" >&2
  exit 1
fi

SEED_INDEX=$(( TASK_ID % NUM_SEEDS ))
ALGO_INDEX=$(( (TASK_ID / NUM_SEEDS) % NUM_ALGOS ))
ENV_INDEX=$(( TASK_ID / (NUM_SEEDS * NUM_ALGOS) ))

ENV_KEY="${ENVS[$ENV_INDEX]}"
ALGO="${ALGOS[$ALGO_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

T_MAX="${T_MAX:-2050000}"
USE_CUDA="${USE_CUDA:-True}"
RESULTS_PATH="${RESULTS_PATH:-results-maco}"

echo "Launching maco task=${TASK_ID}/${TOTAL} env=${ENV_KEY} algo=${ALGO} seed=${SEED}"

python src/main.py --config="${ALGO}" --env-config=maco \
  with env_args.map_name="${ENV_KEY}" seed="${SEED}" t_max="${T_MAX}" use_cuda="${USE_CUDA}" local_results_path="${RESULTS_PATH}"
