#!/usr/bin/env bash
#SBATCH --account=prasanna_933
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-39%32

conda activate gtcg

SEEDS=("0" "1" "2" "3" "4")
ALGOS=("vast" "ltscg")
GYMMA_ENVS=("reference" "speaker" "crypto" "adversary")

declare -A GYMMA_KEYS=(
  ["reference"]="pz-mpe-simple-reference-v3"
  ["speaker"]="pz-mpe-simple-speaker-listener-v4"
  ["crypto"]="pz-mpe-simple-crypto-v3"
  ["adversary"]="pz-mpe-simple-adversary-v3"
)
declare -A GYMMA_WRAPPERS=(
  ["reference"]="-"
  ["speaker"]="-"
  ["crypto"]="-"
  ["adversary"]="PretrainedAdversary"
)

TASKS=()
for algo in "${ALGOS[@]}"; do
  for env in "${GYMMA_ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      TASKS+=("gymma ${env} ${GYMMA_WRAPPERS[$env]} ${algo} ${seed}")
    done
  done
done

TOTAL=${#TASKS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid TASK_ID ${TASK_ID} (TOTAL=${TOTAL})" >&2
  exit 1
fi

read -r ENV_TYPE ENV_NAME WRAPPER ALGO SEED <<< "${TASKS[$TASK_ID]}"

T_MAX="${T_MAX:-2000000}"
TIME_LIMIT="${TIME_LIMIT:-25}"
USE_CUDA="${USE_CUDA:-False}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-gtcg-discovery}"
RESULTS_PATH="${RESULTS_PATH:-results-vast-ltscg-gymma}"

echo "Launching task=${TASK_ID}/${TOTAL} env=${ENV_NAME} algo=${ALGO} seed=${SEED}"

ENV_KEY="${GYMMA_KEYS[$ENV_NAME]}"
WRAPPER_ARG=()
if [[ "${WRAPPER}" != "-" ]]; then
  WRAPPER_ARG=(env_args.pretrained_wrapper="${WRAPPER}")
fi

python src/main.py --config="${ALGO}" --env-config=gymma \
  with env_args.time_limit="${TIME_LIMIT}" env_args.key="${ENV_KEY}" seed="${SEED}" \
  t_max="${T_MAX}" use_cuda="${USE_CUDA}" use_wandb="${USE_WANDB}" \
  wandb_project="${WANDB_PROJECT}" local_results_path="${RESULTS_PATH}" "${WRAPPER_ARG[@]}"
