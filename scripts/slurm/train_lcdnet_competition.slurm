#!/bin/bash

#SBATCH -J lcdnet_comp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A.lcdnet_comp.out

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
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-/data/pjh7639/datasets/raw-train-trainval.zip}"
PRETRAINED_SR_CKPT="${PRETRAINED_SR_CKPT:-/data/pjh7639/datasets/best_model_cgnetV2_deformable_Epoch_82.pth}"
PRETRAINED_OCR_CKPT="${PRETRAINED_OCR_CKPT:-/data/pjh7639/weights/GP_LPR/Rodosol.pth}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/cgnetV2_deformable_competition.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/pjh7639/weights/LCDNet/runs}"

SEED="${SEED:-1996}"
VAL_RATIO="${VAL_RATIO:-0.1}"
ALL_FRAMES="${ALL_FRAMES:-1}"
FRAME_INDEX="${FRAME_INDEX:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCH_MAX="${EPOCH_MAX:-3000}"
BACKGROUND="${BACKGROUND:-(127, 127, 127)}"

STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

RUN_TAG_DEFAULT="lcdnet_comp_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
RUN_OUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/${CONFIG_TEMPLATE}" ]]; then
  echo "[ERROR] config template not found: ${PROJECT_DIR}/${CONFIG_TEMPLATE}"
  exit 1
fi
for f in "${RAW_TRAIN_ZIP}" "${PRETRAINED_SR_CKPT}" "${PRETRAINED_OCR_CKPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] required file not found: ${f}"
    exit 1
  fi
done

mkdir -p "${RUN_OUT_DIR}"

LOCAL_ROOT=""
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
  if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
    echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
    LOCAL_BASE="${HOME}/scratch/pjh7639"
    mkdir -p "${LOCAL_BASE}"
  fi
  LOCAL_ROOT="${LOCAL_BASE}/lcdnet_comp_${RUN_TAG}"
else
  LOCAL_ROOT="${RUN_OUT_DIR}/work"
fi

RAW_WORK_DIR="${LOCAL_ROOT}/raw"
WORK_DIR="${LOCAL_ROOT}/work"
mkdir -p "${RAW_WORK_DIR}" "${WORK_DIR}"

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] raw train zip: ${RAW_TRAIN_ZIP}"
echo "[INFO] output dir: ${RUN_OUT_DIR}"
echo "[INFO] stage_to_local: ${STAGE_TO_LOCAL}"
echo "[INFO] local root: ${LOCAL_ROOT}"
echo "[INFO] val_ratio: ${VAL_RATIO} all_frames: ${ALL_FRAMES} frame_index: ${FRAME_INDEX}"

cp "${RAW_TRAIN_ZIP}" "${RAW_WORK_DIR}/raw-train-trainval.zip"
(
  cd "${RAW_WORK_DIR}"
  unzip -q -o raw-train-trainval.zip
)

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
  local p
  for p in "${parents[@]}"; do
    if has_scenario_pair "${p}"; then
      echo "${p}"
      return 0
    fi
  done

  return 1
}

TRAIN_ROOT="$(resolve_train_root "${RAW_WORK_DIR}" || true)"
if [[ -z "${TRAIN_ROOT}" ]]; then
  echo "[ERROR] failed to resolve train root from extracted zip."
  find "${RAW_WORK_DIR}" -maxdepth 4 -type d | sed -n '1,120p' || true
  exit 1
fi
echo "[INFO] resolved train root: ${TRAIN_ROOT}"

SPLIT_RAW="${WORK_DIR}/split_raw_train.txt"
SPLIT_PREPROC="${WORK_DIR}/split_preprocessed.txt"
PREPROC_ROOT="${WORK_DIR}/preprocessed_16x48_32x96"
RUNTIME_CFG="${WORK_DIR}/cgnetV2_deformable_competition.runtime.yaml"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

SPLIT_ARGS=(
  scripts/preprocess/build_split_from_raw_train.py
  --train-root "${TRAIN_ROOT}"
  --out-split "${SPLIT_RAW}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
)
if [[ "${ALL_FRAMES}" == "1" ]]; then
  SPLIT_ARGS+=(--all-frames)
else
  SPLIT_ARGS+=(--frame-index "${FRAME_INDEX}")
fi
"${PY_RUN[@]}" "${SPLIT_ARGS[@]}"

"${PY_RUN[@]}" scripts/preprocess/preprocess_parallel_split.py \
  --split-in "${SPLIT_RAW}" \
  --out-root "${PREPROC_ROOT}" \
  --split-out "${SPLIT_PREPROC}" \
  --lr-w 48 \
  --lr-h 16 \
  --background "${BACKGROUND}"

"${PY_RUN[@]}" - <<PY
import yaml
from pathlib import Path

template = Path("${PROJECT_DIR}") / "${CONFIG_TEMPLATE}"
runtime = Path("${RUNTIME_CFG}")
cfg = yaml.safe_load(template.read_text(encoding="utf-8"))

cfg["MODEL_OCR"]["OCR_TRAIN"] = False
cfg["epoch_max"] = int("${EPOCH_MAX}")
cfg["resume"] = None
cfg["LOAD_PRE_TRAINED_SR"] = "${PRETRAINED_SR_CKPT}"
cfg["LOAD_PRE_TRAINED_OCR"] = "${PRETRAINED_OCR_CKPT}"

cfg["train_dataset"]["dataset"]["args"]["path_split"] = "${SPLIT_PREPROC}"
cfg["train_dataset"]["batch"] = int("${BATCH_SIZE}")
cfg["train_dataset"]["num_workers"] = int("${NUM_WORKERS}")

cfg["val_dataset"]["dataset"]["args"]["path_split"] = "${SPLIT_PREPROC}"
cfg["val_dataset"]["batch"] = int("${BATCH_SIZE}")
cfg["val_dataset"]["num_workers"] = int("${NUM_WORKERS}")

runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[INFO] runtime config saved: {runtime}")
PY

"${PY_RUN[@]}" ParallelNetTrain.py --config "${RUNTIME_CFG}" --save "${RUN_TAG}"

SAVE_DIR="${PROJECT_DIR}/save/${RUN_TAG}"
if [[ ! -d "${SAVE_DIR}" ]]; then
  echo "[ERROR] expected save dir not found: ${SAVE_DIR}"
  exit 1
fi

mkdir -p "${RUN_OUT_DIR}"
cp -a "${SAVE_DIR}" "${RUN_OUT_DIR}/"
cp -f "${RUNTIME_CFG}" "${RUN_OUT_DIR}/"
cp -f "${SPLIT_RAW}" "${RUN_OUT_DIR}/"
cp -f "${SPLIT_PREPROC}" "${RUN_OUT_DIR}/"

echo "[DONE] LCDNet competition training finished."
echo "[DONE] run tag: ${RUN_TAG}"
echo "[DONE] output dir: ${RUN_OUT_DIR}"
