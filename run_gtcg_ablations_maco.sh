#!/usr/bin/env bash
set -euo pipefail

# GTCG ablations on MACO maps.
#
# 4 ablation families implemented in this script:
# 1) Multi-head: transf, H=1..4, L fixed, PM0
# 2) Conv swap: transf/gcn/gat, H fixed, L fixed, PM0
# 3) Multi-hop: transf, L=1..3, H fixed, PM0
# 4) Parameter-control: transf (PM0 reference) + mlp/gcn/gat (PM1)
#
# Quick dry-run (single command preview):
# DRY_RUN=1 FAMILY=multihead ENVS_OVERRIDE="disperse" SEEDS_OVERRIDE="0" bash run_gtcg_ablations_maco.sh
#
# Quick real run (single setting):
# FAMILY=paramctrl ENVS_OVERRIDE="disperse" SEEDS_OVERRIDE="0" PARAMCTRL_HEADS=2 PARAMCTRL_LAYERS=2 MAX_RUNS=4 bash run_gtcg_ablations_maco.sh

ENVS=(${ENVS_OVERRIDE:-aloha disperse gather hallway pursuit sensor})
SEEDS=(${SEEDS_OVERRIDE:-0})

T_MAX="${T_MAX:-2050000}"
USE_CUDA="${USE_CUDA:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-gtcg-ablations}"
RESULTS_PATH="${RESULTS_PATH:-results-maco}"
FAMILY="${FAMILY:-all}" # one of: all, multihead, convswap, multihop, paramctrl
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}" # 0 means unlimited

MULTIHEAD_LAYERS="${MULTIHEAD_LAYERS:-2}"
CONVSWAP_HEADS="${CONVSWAP_HEADS:-2}"
CONVSWAP_LAYERS="${CONVSWAP_LAYERS:-2}"
MULTIHOP_HEADS="${MULTIHOP_HEADS:-2}"
PARAMCTRL_HEADS="${PARAMCTRL_HEADS:-2}"
PARAMCTRL_LAYERS="${PARAMCTRL_LAYERS:-2}"

RUN_COUNT=0

run_one() {
  local conv="$1"
  local heads="$2"
  local layers="$3"
  local pm_int="$4"
  local env_key="$5"
  local seed="$6"

  if [[ "${MAX_RUNS}" != "0" ]] && (( RUN_COUNT >= MAX_RUNS )); then
    return 1
  fi

  local pm_bool="False"
  if [[ "${pm_int}" == "1" ]]; then
    pm_bool="True"
  fi
  local run_tag="abl_${conv}_H${heads}_L${layers}_PM${pm_int}"

  local cmd=(
    python src/main.py --config=gtcg --env-config=maco
    with
    env_args.map_name="${env_key}"
    seed="${seed}"
    t_max="${T_MAX}"
    use_cuda="${USE_CUDA}"
    wandb_project="${WANDB_PROJECT}"
    local_results_path="${RESULTS_PATH}"
    gtcg_conv_type="${conv}"
    transformer_heads="${heads}"
    number_gcn_layers="${layers}"
    param_match="${pm_bool}"
    run_tag="${run_tag}"
  )

  echo "Launching env=${env_key} seed=${seed} tag=${run_tag}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%q " "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi

  RUN_COUNT=$((RUN_COUNT + 1))
  return 0
}

run_multihead() {
  for env_key in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for heads in 1 2 3 4; do
        run_one "transf" "${heads}" "${MULTIHEAD_LAYERS}" "0" "${env_key}" "${seed}" || return 1
      done
    done
  done
}

run_convswap() {
  for env_key in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for conv in transf gcn gat; do
        run_one "${conv}" "${CONVSWAP_HEADS}" "${CONVSWAP_LAYERS}" "0" "${env_key}" "${seed}" || return 1
      done
    done
  done
}

run_multihop() {
  for env_key in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for layers in 1 2 3; do
        run_one "transf" "${MULTIHOP_HEADS}" "${layers}" "0" "${env_key}" "${seed}" || return 1
      done
    done
  done
}

run_paramctrl() {
  for env_key in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "transf" "${PARAMCTRL_HEADS}" "${PARAMCTRL_LAYERS}" "0" "${env_key}" "${seed}" || return 1
      for conv in mlp gcn gat; do
        run_one "${conv}" "${PARAMCTRL_HEADS}" "${PARAMCTRL_LAYERS}" "1" "${env_key}" "${seed}" || return 1
      done
    done
  done
}

case "${FAMILY}" in
  all)
    run_multihead || true
    run_convswap || true
    run_multihop || true
    run_paramctrl || true
    ;;
  multihead)
    run_multihead
    ;;
  convswap)
    run_convswap
    ;;
  multihop)
    run_multihop
    ;;
  paramctrl)
    run_paramctrl
    ;;
  *)
    echo "Invalid FAMILY=${FAMILY}. Use one of: all, multihead, convswap, multihop, paramctrl" >&2
    exit 1
    ;;
esac

echo "Completed runs: ${RUN_COUNT}"
