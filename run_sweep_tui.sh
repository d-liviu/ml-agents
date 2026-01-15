#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-mlagents310fresh}"
PYTHON_RUNNER=(conda run -n "${ENV_NAME}" python)

ASCII_ART='
 __  __ _       ____            _           _   
|  \/  | |     |  _ \ _ __ ___  (_) ___  ___| |_ 
| |\/| | |_____| |_) | '"'"'__/ _ \ | |/ _ \/ __| __|
| |  | | |_____|  __/| | | (_) || |  __/ (__| |_ 
|_|  |_|_|     |_|   |_|  \___(_)_|\___|\___|\__|


⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣶⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣄⣀⡀⣠⣾⡇⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⢿⣿⣿⡇⠀⠀⠀⠀
⠀⣶⣿⣦⣜⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⡿⢿⡏⣴⣺⣦⣙⣿⣷⣄⠀⠀⠀
⠀⣯⡇⣻⣿⣿⣿⣿⣷⣾⣿⣬⣥⣭⣽⣿⣿⣧⣼⡇⣯⣇⣹⣿⣿⣿⣿⣧⠀⠀
⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠸⣿⣿⣿⣿⣿⣿⣿⣷⠀
'
EXIT_ART='
⠟⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠛⢻⣿
⡆⠊⠈⣿⢿⡟⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣎⠈⠻
⣷⣠⠁⢀⠰⠀⣰⣿⣿⣿⣿⣿⣿⠟⠋⠛⠛⠿⠿⢿⣿⣿⣿⣧⠀⢹⣿⡑⠐⢰
⣿⣿⠀⠁⠀⠀⣿⣿⣿⣿⠟⡩⠐⠀⠀⠀⠀⢐⠠⠈⠊⣿⣿⣿⡇⠘⠁⢀⠆⢀
⣿⣿⣆⠀⠀⢤⣿⣿⡿⠃⠈⠀⣠⣶⣿⣿⣷⣦⡀⠀⠀⠈⢿⣿⣇⡆⠀⠀⣠⣾
⣿⣿⣿⣧⣦⣿⣿⣿⡏⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠐⣿⣿⣷⣦⣷⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⣾⣿⣿⠋⠁⠀⠉⠻⣿⣿⣧⠀⠠⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⣿⡿⠁⠀⠀⠀⠀⠀⠘⢿⣿⠀⣺⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣠⣂⠀⠀⠀⠀⠀⠀⠀⢀⣁⢠⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣄⣤⣤⣔⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
'

echo "${ASCII_ART}"
echo "ML-Agents Random Sweep TUI"
echo "Env: ${ENV_NAME}"
echo

prompt() {
  local label="$1"
  local default="${2:-}"
  local value
  if [[ -n "${default}" ]]; then
    read -r -p "${label} [${default}]: " value
    value="${value:-${default}}"
  else
    read -r -p "${label}: " value
  fi
  echo "${value}"
}

choose_env() {
  echo "Select environment:"
  echo "  1) Worm (auto defaults)"
  echo "  2) Pyramids (auto defaults)"
  echo "  3) Custom (manual paths)"
  local choice
  read -r -p "Choice (1=Worm, 2=Pyramids, 3=Custom): " choice
  case "${choice}" in
    1) echo "Worm" ;;
    2) echo "Pyramids" ;;
    3) echo "Custom" ;;
    *) echo "Custom" ;;
  esac
}

run_sweep() {
  local mode="$1"
  local env_choice
  env_choice="$(choose_env)"

  local env_path base_yaml out_dir prefix
  local n_configs start_index count seeds max_steps

  if [[ "${env_choice}" == "Worm" ]]; then
    env_path="$(prompt "Unity app path" "Project/Worm.app")"
    base_yaml="$(prompt "Base YAML" "config/ppo/Worm/Worm.yaml")"
    out_dir="$(prompt "Out dir" "results/sweep_worm")"
    prefix="$(prompt "Run prefix" "WormA")"
    max_steps="$(prompt "Max steps" "2000000")"
  elif [[ "${env_choice}" == "Pyramids" ]]; then
    env_path="$(prompt "Unity app path" "Project/Pyramids.app")"
    base_yaml="$(prompt "Base YAML" "config/ppo/Pyramids/Pyramids.yaml")"
    out_dir="$(prompt "Out dir" "results/sweep_pyramids")"
    prefix="$(prompt "Run prefix" "PyrA")"
    max_steps="$(prompt "Max steps" "1000000")"
  else
    env_path="$(prompt "Unity app path")"
    base_yaml="$(prompt "Base YAML")"
    out_dir="$(prompt "Out dir" "results/sweep_custom")"
    prefix="$(prompt "Run prefix" "Sweep")"
    max_steps="$(prompt "Max steps" "1000000")"
  fi

  n_configs="$(prompt "Total configs in sweep" "70")"
  start_index="$(prompt "Start index" "0")"
  count="$(prompt "Count" "10")"
  seeds="$(prompt "Seeds (comma-separated)" "0")"

  local args=(
    "run_random_sweep.py"
    "--env-path" "${env_path}"
    "--base-yaml" "${base_yaml}"
    "--out-dir" "${out_dir}"
    "--prefix" "${prefix}"
    "--n-configs" "${n_configs}"
    "--start-index" "${start_index}"
    "--count" "${count}"
    "--seeds" "${seeds}"
    "--max-steps" "${max_steps}"
  )

  if [[ "${mode}" == "generate" ]]; then
    args+=("--generate-only")
  else
    args+=("--force")
  fi

  echo
  echo "Running: ${PYTHON_RUNNER[*]} ${args[*]}"
  echo
  MLAGENTS_SKIP_ONNX_EXPORT=1 "${PYTHON_RUNNER[@]}" "${args[@]}"
}

while true; do
  echo
  echo "Menu:"
  echo "  1) Randomize configs (generate only)"
  echo "  2) Run training"
  echo "  3) Exit"
  read -r -p "Select: " action
  case "${action}" in
    1) run_sweep "generate" ;;
    2) run_sweep "train" ;;
    3)
      echo "${EXIT_ART}"
      exit 0
      ;;
    *) echo "Unknown option." ;;
  esac
done
