#!/bin/bash

#SBATCH -J gplpr_2stage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A.gplpr_2stage.out

set -euo pipefail

source ~/.bashrc || true

if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "${HOME}/miniconda3/bin/conda" ]]; then
    export PATH="${HOME}/miniconda3/bin:${PATH}"
  elif [[ -f "${HOME}/anaconda3/bin/conda" ]]; then
    export PATH="${HOME}/anaconda3/bin:${PATH}"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda binary not found in this batch environment."
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "googlenet"; then
  echo "[ERROR] conda env not found: googlenet"
  conda env list || true
  exit 1
fi

PY_RUN=(conda run --no-capture-output -n googlenet python -u)

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/pjh7639/weights/GP_LPR}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/data/pjh7639/weights/GP_LPR/Rodosol.pth}"

TRAIN_ROOT_NAS="${TRAIN_ROOT_NAS:-/data/pjh7639/datasets/raw_train_unzip}"
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-/data/pjh7639/datasets/raw-train-trainval.zip}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"     # 1: local SSD staging
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"   # 1: keep local staged data after run
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

SEED="${SEED:-1996}"
VAL_RATIO="${VAL_RATIO:-0.1}"
STAGE1_A_TRACKS="${STAGE1_A_TRACKS:-9000}"
STAGE1_B_TRACKS="${STAGE1_B_TRACKS:-4500}"
STAGE2_B_TRACKS="${STAGE2_B_TRACKS:-4500}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-40}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ALL_FRAMES="${ALL_FRAMES:-1}"
FRAME_INDEX="${FRAME_INDEX:-3}"
IMG_W="${IMG_W:-48}"
IMG_H="${IMG_H:-16}"
BACKGROUND="${BACKGROUND:-(127, 127, 127)}"
PREPROCESSED="${PREPROCESSED:-0}"
DRY_RUN_COLLECT="${DRY_RUN_COLLECT:-0}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gplpr-curriculum}"
WANDB_ENTITY="${WANDB_ENTITY:-pyojihun80-kyung-hee-university}"
WANDB_MODE="${WANDB_MODE:-online}"   # online | offline | disabled
WANDB_DIR="${WANDB_DIR:-}"

RUN_TAG_DEFAULT="gplpr_2stage_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
RUN_DIR="${RUN_DIR:-${WEIGHTS_ROOT}/runs/${RUN_TAG}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "[ERROR] PRETRAINED_CKPT not found: ${PRETRAINED_CKPT}"
  exit 1
fi

mkdir -p "${WEIGHTS_ROOT}" "${RUN_DIR}"
cd "${PROJECT_DIR}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] WANDB_API_KEY detected. Logging in to W&B."
  conda run --no-capture-output -n googlenet wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "[WARN] WANDB_API_KEY is not set. W&B online mode may fail."
fi

EFFECTIVE_TRAIN_ROOT="${TRAIN_ROOT_NAS}"
LOCAL_ROOT=""

has_scenario_pair() {
  local base="$1"
  local a_dir b_dir
  a_dir="$(find "${base}" -maxdepth 1 -mindepth 1 -type d \( -iname 'Scenario-A' -o -iname 'Senario-A' \) | head -n1 || true)"
  b_dir="$(find "${base}" -maxdepth 1 -mindepth 1 -type d \( -iname 'Scenario-B' -o -iname 'Senario-B' \) | head -n1 || true)"
  [[ -n "${a_dir}" && -n "${b_dir}" ]]
}

resolve_train_root() {
  local base="$1"
  if [[ ! -d "${base}" ]]; then
    return 1
  fi

  if has_scenario_pair "${base}"; then
    echo "${base}"
    return 0
  fi

  if [[ -d "${base}/train" ]] && has_scenario_pair "${base}/train"; then
    echo "${base}/train"
    return 0
  fi

  mapfile -t parents < <(
    find "${base}" -type d \( -iname 'Scenario-A' -o -iname 'Senario-A' -o -iname 'Scenario-B' -o -iname 'Senario-B' \) \
      -exec dirname {} \; | sort -u
  )

  local best=""
  local best_depth=999999
  local p depth
  for p in "${parents[@]}"; do
    if has_scenario_pair "${p}"; then
      depth=$(awk -F/ '{print NF}' <<< "${p}")
      if (( depth < best_depth )); then
        best="${p}"
        best_depth=${depth}
      fi
    fi
  done

  if [[ -n "${best}" ]]; then
    echo "${best}"
    return 0
  fi

  return 1
}

cleanup_local() {
  if [[ "${KEEP_LOCAL_DATA}" == "1" ]]; then
    return
  fi
  if [[ -n "${LOCAL_ROOT}" && -d "${LOCAL_ROOT}" ]]; then
    rm -rf "${LOCAL_ROOT}" || true
  fi
}
trap cleanup_local EXIT

if [[ "${STAGE_TO_LOCAL}" == "1" ]]; then
  if [[ ! -f "${RAW_TRAIN_ZIP}" ]]; then
    echo "[WARN] STAGE_TO_LOCAL=1 but RAW_TRAIN_ZIP not found: ${RAW_TRAIN_ZIP}"
    echo "[WARN] fallback to TRAIN_ROOT_NAS=${TRAIN_ROOT_NAS}"
  else
    if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
      echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
      LOCAL_BASE="${HOME}/scratch/pjh7639"
      mkdir -p "${LOCAL_BASE}"
    fi

    LOCAL_ROOT="${LOCAL_BASE}/gplpr_2stage_${RUN_TAG}"
    LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
    mkdir -p "${LOCAL_RAW_DIR}"

    echo "[INFO] staging raw-train.zip to local SSD: ${LOCAL_ROOT}"
    cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
    (
      cd "${LOCAL_RAW_DIR}"
      unzip -q -o "$(basename "${RAW_TRAIN_ZIP}")"
    )
    EFFECTIVE_TRAIN_ROOT="${LOCAL_RAW_DIR}"
  fi
fi

RESOLVED_TRAIN_ROOT="$(resolve_train_root "${EFFECTIVE_TRAIN_ROOT}" || true)"
if [[ -z "${RESOLVED_TRAIN_ROOT}" ]]; then
  echo "[ERROR] failed to resolve train root with Scenario-A/B pair from: ${EFFECTIVE_TRAIN_ROOT}"
  echo "[INFO] scenario-like dirs:"
  find "${EFFECTIVE_TRAIN_ROOT}" -type d | grep -Ei 'scenario|senario' | sed -n '1,80p' || true
  exit 1
fi
EFFECTIVE_TRAIN_ROOT="${RESOLVED_TRAIN_ROOT}"

if [[ ! -d "${EFFECTIVE_TRAIN_ROOT}" ]]; then
  echo "[ERROR] train root not found: ${EFFECTIVE_TRAIN_ROOT}"
  exit 1
fi

echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] effective train root: ${EFFECTIVE_TRAIN_ROOT}"
echo "[INFO] pretrained ckpt: ${PRETRAINED_CKPT}"
echo "[INFO] run dir: ${RUN_DIR}"
echo "[INFO] weights root: ${WEIGHTS_ROOT}"
echo "[INFO] val ratio: ${VAL_RATIO}"
echo "[INFO] stage1 tracks: A=${STAGE1_A_TRACKS}, B=${STAGE1_B_TRACKS}"
echo "[INFO] stage2 tracks: B=${STAGE2_B_TRACKS}"
echo "[INFO] stage epochs: stage1=${STAGE1_EPOCHS}, stage2=${STAGE2_EPOCHS}"

SCENARIO_A_DIR="$(find "${EFFECTIVE_TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d \( -iname 'Scenario-A' -o -iname 'Senario-A' \) | head -n1 || true)"
SCENARIO_B_DIR="$(find "${EFFECTIVE_TRAIN_ROOT}" -maxdepth 1 -mindepth 1 -type d \( -iname 'Scenario-B' -o -iname 'Senario-B' \) | head -n1 || true)"
SAMPLE_TRACK_COUNT="$(find "${EFFECTIVE_TRAIN_ROOT}" -type d \( -iname 'track_*' -o -iname 'track-*' \) | wc -l | tr -d ' ')"
echo "[INFO] preflight Scenario-A dir: ${SCENARIO_A_DIR:-<missing>}"
echo "[INFO] preflight Scenario-B dir: ${SCENARIO_B_DIR:-<missing>}"
echo "[INFO] preflight sample track count: ${SAMPLE_TRACK_COUNT}"
if [[ -z "${SCENARIO_A_DIR}" || -z "${SCENARIO_B_DIR}" ]]; then
  echo "[ERROR] Scenario-A/B directories were not both detected under: ${EFFECTIVE_TRAIN_ROOT}"
  exit 1
fi

ARGS=(
  train_gplpr_2staged.py
  --train-root "${EFFECTIVE_TRAIN_ROOT}"
  --pretrained-ckpt "${PRETRAINED_CKPT}"
  --output-dir "${RUN_DIR}"
  --seed "${SEED}"
  --val-ratio "${VAL_RATIO}"
  --stage1-a-tracks "${STAGE1_A_TRACKS}"
  --stage1-b-tracks "${STAGE1_B_TRACKS}"
  --stage2-b-tracks "${STAGE2_B_TRACKS}"
  --all-frames "${ALL_FRAMES}"
  --frame-index "${FRAME_INDEX}"
  --dry-run-collect "${DRY_RUN_COLLECT}"
  --img-w "${IMG_W}"
  --img-h "${IMG_H}"
  --background "${BACKGROUND}"
  --preprocessed "${PREPROCESSED}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --wandb-enabled "${WANDB_ENABLED}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-run-name "${WANDB_RUN_NAME}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_DIR}" ]]; then
  ARGS+=(--wandb-dir "${WANDB_DIR}")
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
"${PY_RUN[@]}" "${ARGS[@]}"

echo "[INFO] copying checkpoints to ${WEIGHTS_ROOT}"
for src in \
  "${RUN_DIR}/stage1/best_stage1.pth" \
  "${RUN_DIR}/stage1/last_stage1.pth" \
  "${RUN_DIR}/stage2/best_stage2.pth" \
  "${RUN_DIR}/stage2/last_stage2.pth"; do
  if [[ -f "${src}" ]]; then
    dst="${WEIGHTS_ROOT}/${RUN_TAG}_$(basename "${src}")"
    cp -f "${src}" "${dst}"
    echo "[INFO] saved: ${dst}"
  else
    echo "[WARN] checkpoint missing: ${src}"
  fi
done

if [[ -f "${WEIGHTS_ROOT}/${RUN_TAG}_best_stage2.pth" ]]; then
  ln -sfn "${WEIGHTS_ROOT}/${RUN_TAG}_best_stage2.pth" "${WEIGHTS_ROOT}/latest_gplpr_stage2_best.pth"
fi
if [[ -f "${WEIGHTS_ROOT}/${RUN_TAG}_best_stage1.pth" ]]; then
  ln -sfn "${WEIGHTS_ROOT}/${RUN_TAG}_best_stage1.pth" "${WEIGHTS_ROOT}/latest_gplpr_stage1_best.pth"
fi

for f in "${RUN_DIR}/run_summary.json" "${RUN_DIR}/track_selection.json"; do
  if [[ -f "${f}" ]]; then
    cp -f "${f}" "${WEIGHTS_ROOT}/${RUN_TAG}_$(basename "${f}")"
  fi
done

echo "[DONE] GPLPR 2-stage curriculum training completed."
echo "[DONE] run dir: ${RUN_DIR}"
