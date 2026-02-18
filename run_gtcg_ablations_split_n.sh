#!/usr/bin/env bash
#SBATCH --account=prasanna_1363
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-329%32

conda activate gtcg

SEEDS=("0" "1" "2" "3" "4")

# Split 1 (~50%): 6 / 12 envs
MACO_ENVS=("gather" "hallway")
GYMMA_ENVS=("reference" "tag" "crypto" "worldcomm")

declare -A GYMMA_KEYS=(
  ["reference"]="pz-mpe-simple-reference-v3"
  ["tag"]="pz-mpe-simple-tag-v3"
  ["crypto"]="pz-mpe-simple-crypto-v3"
  ["worldcomm"]="pz-mpe-simple-world-comm-v3"
  ["spread"]="pz-mpe-simple-spread-v3"
  ["push"]="pz-mpe-simple-push-v3"
  ["adversary"]="pz-mpe-simple-adversary-v3"
  ["speaker-listener"]="pz-mpe-simple-speaker-listener-v4"
)
declare -A GYMMA_WRAPPERS=(
  ["reference"]="-"
  ["tag"]="PretrainedTag"
  ["crypto"]="-"
  ["worldcomm"]="-"
  ["spread"]="-"
  ["push"]="-"
  ["adversary"]="PretrainedAdversary"
  ["speaker-listener"]="-"
)

# Unique ablation grid (covers all 4 families without duplicate identical runs):
# multi-head: transf H={1,2,3,4}, L=2, PM0
# conv swap: gcn/gat/transf at H=2, L=2, PM0
# multi-hop: transf L={1,2,3}, H=2, PM0
# parameter-control: mlp/gcn/gat at H=2, L=2, PM1 (+ transf H2 L2 PM0 as reference)
ABLATIONS=(
  "transf 1 2 0"
  "transf 2 2 0"
  "transf 3 2 0"
  "transf 4 2 0"
  "gcn 2 2 0"
  "gat 2 2 0"
  "transf 2 1 0"
  "transf 2 3 0"
  "mlp 2 2 1"
  "gcn 2 2 1"
  "gat 2 2 1"
)

TASKS=()
for env in "${MACO_ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
      read -r conv heads layers pm <<< "${ablation}"
      TASKS+=("maco ${env} - ${seed} ${conv} ${heads} ${layers} ${pm}")
    done
  done
done

for env in "${GYMMA_ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ablation in "${ABLATIONS[@]}"; do
      read -r conv heads layers pm <<< "${ablation}"
      TASKS+=("gymma ${env} ${GYMMA_WRAPPERS[$env]} ${seed} ${conv} ${heads} ${layers} ${pm}")
    done
  done
done

TOTAL=${#TASKS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid TASK_ID ${TASK_ID} (TOTAL=${TOTAL})" >&2
  exit 1
fi

read -r ENV_TYPE ENV_NAME WRAPPER SEED CONV HEADS LAYERS PM <<< "${TASKS[$TASK_ID]}"

T_MAX="${T_MAX:-2050000}"
TIME_LIMIT="${TIME_LIMIT:-25}"
USE_CUDA="${USE_CUDA:-False}"
USE_WANDB="${USE_WANDB:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-gtcg-ablations}"
RESULTS_PATH="${RESULTS_PATH:-results-gtcg-ablations-split1}"

PM_BOOL="False"
if [[ "${PM}" == "1" ]]; then
  PM_BOOL="True"
fi
RUN_TAG="abl_${CONV}_H${HEADS}_L${LAYERS}_PM${PM}"

echo "Launching task=${TASK_ID}/${TOTAL} env_type=${ENV_TYPE} env=${ENV_NAME} seed=${SEED} tag=${RUN_TAG}"

if [[ "${ENV_TYPE}" == "maco" ]]; then
  python src/main.py --config=gtcg --env-config=maco \
    with env_args.map_name="${ENV_NAME}" seed="${SEED}" t_max="${T_MAX}" \
    use_cuda="${USE_CUDA}" use_wandb="${USE_WANDB}" wandb_project="${WANDB_PROJECT}" \
    local_results_path="${RESULTS_PATH}" gtcg_conv_type="${CONV}" transformer_heads="${HEADS}" \
    number_gcn_layers="${LAYERS}" param_match="${PM_BOOL}" run_tag="${RUN_TAG}"
else
  ENV_KEY="${GYMMA_KEYS[$ENV_NAME]}"
  WRAPPER_ARG=()
  if [[ "${WRAPPER}" != "-" ]]; then
    WRAPPER_ARG=(env_args.pretrained_wrapper="${WRAPPER}")
  fi
  python src/main.py --config=gtcg --env-config=gymma \
    with env_args.time_limit="${TIME_LIMIT}" env_args.key="${ENV_KEY}" seed="${SEED}" \
    t_max="${T_MAX}" use_cuda="${USE_CUDA}" use_wandb="${USE_WANDB}" \
    wandb_project="${WANDB_PROJECT}" local_results_path="${RESULTS_PATH}" \
    gtcg_conv_type="${CONV}" transformer_heads="${HEADS}" number_gcn_layers="${LAYERS}" \
    param_match="${PM_BOOL}" run_tag="${RUN_TAG}" "${WRAPPER_ARG[@]}"
fi
