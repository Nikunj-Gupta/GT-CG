#!/usr/bin/env bash
#SBATCH --account=prasanna_933
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:00:05
#SBATCH --array=0-79

conda activate gtcg

ENVS=(
  "pz-mpe-simple-adversary-v3"
  "pz-mpe-simple-speaker-listener-v4"
  "pz-mpe-simple-world-comm-v3"
  "pz-mpe-simple-crypto-v3"
  "pz-mpe-simple-spread-v3"
  "pz-mpe-simple-push-v3"
  "pz-mpe-simple-tag-v3"
  "pz-mpe-simple-reference-v3"
)
WRAPPERS=(
  "PretrainedAdversary"
  ""
  ""
  ""
  ""
  ""
  "PretrainedTag"
  ""
)
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
WRAPPER="${WRAPPERS[$ENV_INDEX]}"
ALGO="${ALGOS[$ALGO_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

T_MAX="${T_MAX:-25000}"
TIME_LIMIT="${TIME_LIMIT:-25}"
USE_CUDA="${USE_CUDA:-True}"
RESULTS_PATH="${RESULTS_PATH:-results-gymma}"

echo "Launching gymma task=${TASK_ID}/${TOTAL} env=${ENV_KEY} algo=${ALGO} seed=${SEED}"

WRAPPER_ARG=()
if [[ -n "${WRAPPER}" ]]; then
  WRAPPER_ARG=(with env_args.pretrained_wrapper="${WRAPPER}")
fi

python src/main.py --config="${ALGO}" --env-config=gymma \
  with env_args.time_limit="${TIME_LIMIT}" env_args.key="${ENV_KEY}" seed="${SEED}" \
  t_max="${T_MAX}" use_cuda="${USE_CUDA}" local_results_path="${RESULTS_PATH}" \
  "${WRAPPER_ARG[@]}"
