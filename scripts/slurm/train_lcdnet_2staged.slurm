#!/bin/bash

#SBATCH -J lcdnet_2stage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A.lcdnet_2stage.out

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
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/pjh7639/weights/LCDNet/runs}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

PRETRAINED_SR_CKPT="${PRETRAINED_SR_CKPT:-/data/pjh7639/datasets/best_model_cgnetV2_deformable_Epoch_82.pth}"
PRETRAINED_OCR_CKPT="${PRETRAINED_OCR_CKPT:-/data/pjh7639/weights/GP_LPR/gplpr_2stage_66324_20260220_033315_last_stage2.pth}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/cgnetV2_deformable_competition.yaml}"

STAGE1_A_TRACKS="${STAGE1_A_TRACKS:-9000}"
STAGE1_B_TRACKS="${STAGE1_B_TRACKS:-4500}"
STAGE2_B_TRACKS="${STAGE2_B_TRACKS:-4500}"
VAL_RATIO="${VAL_RATIO:-0.1}"
ALL_FRAMES="${ALL_FRAMES:-1}"
FRAME_INDEX="${FRAME_INDEX:-3}"
SEED="${SEED:-1996}"

BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE1_EPOCH_MAX="${STAGE1_EPOCH_MAX:-40}"
STAGE2_EPOCH_MAX="${STAGE2_EPOCH_MAX:-20}"
BACKGROUND="${BACKGROUND:-(127, 127, 127)}"

RUN_TAG_DEFAULT="lcdnet_2stage_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"

STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"
DRY_RUN_COLLECT="${DRY_RUN_COLLECT:-0}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/${CONFIG_TEMPLATE}" ]]; then
  echo "[ERROR] config template not found: ${PROJECT_DIR}/${CONFIG_TEMPLATE}"
  exit 1
fi
if [[ ! -f "${RAW_TRAIN_ZIP}" ]]; then
  echo "[ERROR] RAW_TRAIN_ZIP not found: ${RAW_TRAIN_ZIP}"
  exit 1
fi
for f in "${PRETRAINED_SR_CKPT}" "${PRETRAINED_OCR_CKPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] required checkpoint not found: ${f}"
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}"

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
  LOCAL_ROOT="${LOCAL_BASE}/lcdnet_2stage_${RUN_TAG}"
else
  LOCAL_ROOT="${RUN_ROOT}/work_local"
fi

RAW_WORK_DIR="${LOCAL_ROOT}/raw"
WORK_DIR="${LOCAL_ROOT}/work"
mkdir -p "${RAW_WORK_DIR}" "${WORK_DIR}"

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] run tag: ${RUN_TAG}"
echo "[INFO] run root: ${RUN_ROOT}"
echo "[INFO] raw train zip: ${RAW_TRAIN_ZIP}"
echo "[INFO] pretrained SR ckpt: ${PRETRAINED_SR_CKPT}"
echo "[INFO] pretrained OCR ckpt: ${PRETRAINED_OCR_CKPT}"
echo "[INFO] stage_to_local: ${STAGE_TO_LOCAL}"
echo "[INFO] local root: ${LOCAL_ROOT}"
echo "[INFO] stage1 tracks: A=${STAGE1_A_TRACKS}, B=${STAGE1_B_TRACKS}"
echo "[INFO] stage2 tracks: B=${STAGE2_B_TRACKS}"

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

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

SPLIT_OUT_ROOT="${RUN_ROOT}/split_seed_${SEED}"
mkdir -p "${SPLIT_OUT_ROOT}"

COLLECT_ARGS=(
  train_gplpr_2staged.py
  --train-root "${TRAIN_ROOT}"
  --pretrained-ckpt "${PRETRAINED_OCR_CKPT}"
  --output-dir "${SPLIT_OUT_ROOT}"
  --seed "${SEED}"
  --val-ratio "${VAL_RATIO}"
  --stage1-a-tracks "${STAGE1_A_TRACKS}"
  --stage1-b-tracks "${STAGE1_B_TRACKS}"
  --stage2-b-tracks "${STAGE2_B_TRACKS}"
  --all-frames "${ALL_FRAMES}"
  --frame-index "${FRAME_INDEX}"
  --dry-run-collect 1
)
"${PY_RUN[@]}" "${COLLECT_ARGS[@]}"

STAGE1_RAW_SPLIT="${SPLIT_OUT_ROOT}/splits/stage1_ab_split.txt"
STAGE2_RAW_SPLIT="${SPLIT_OUT_ROOT}/splits/stage2_b_split.txt"

for f in "${STAGE1_RAW_SPLIT}" "${STAGE2_RAW_SPLIT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] expected split file missing: ${f}"
    exit 1
  fi
done

STAGE1_PREPROC_ROOT="${WORK_DIR}/stage1_preprocessed_16x48_32x96"
STAGE2_PREPROC_ROOT="${WORK_DIR}/stage2_preprocessed_16x48_32x96"
STAGE1_PREPROC_SPLIT="${WORK_DIR}/split_stage1_preprocessed.txt"
STAGE2_PREPROC_SPLIT="${WORK_DIR}/split_stage2_preprocessed.txt"

"${PY_RUN[@]}" scripts/preprocess/preprocess_parallel_split.py \
  --split-in "${STAGE1_RAW_SPLIT}" \
  --out-root "${STAGE1_PREPROC_ROOT}" \
  --split-out "${STAGE1_PREPROC_SPLIT}" \
  --lr-w 48 \
  --lr-h 16 \
  --background "${BACKGROUND}"

"${PY_RUN[@]}" scripts/preprocess/preprocess_parallel_split.py \
  --split-in "${STAGE2_RAW_SPLIT}" \
  --out-root "${STAGE2_PREPROC_ROOT}" \
  --split-out "${STAGE2_PREPROC_SPLIT}" \
  --lr-w 48 \
  --lr-h 16 \
  --background "${BACKGROUND}"

if [[ "${DRY_RUN_COLLECT}" == "1" ]]; then
  echo "[DONE] dry-run collect completed."
  echo "[DONE] stage1 raw split: ${STAGE1_RAW_SPLIT}"
  echo "[DONE] stage2 raw split: ${STAGE2_RAW_SPLIT}"
  echo "[DONE] stage1 preprocessed split: ${STAGE1_PREPROC_SPLIT}"
  echo "[DONE] stage2 preprocessed split: ${STAGE2_PREPROC_SPLIT}"
  exit 0
fi

STAGE1_RUNTIME_CFG="${WORK_DIR}/cgnetV2_deformable.stage1.runtime.yaml"
STAGE2_RUNTIME_CFG="${WORK_DIR}/cgnetV2_deformable.stage2.runtime.yaml"
STAGE1_SAVE_TAG="${RUN_TAG}_stage1"
STAGE2_SAVE_TAG="${RUN_TAG}_stage2"
STAGE1_SAVE_DIR="${PROJECT_DIR}/save/${STAGE1_SAVE_TAG}"
STAGE2_SAVE_DIR="${PROJECT_DIR}/save/${STAGE2_SAVE_TAG}"

make_runtime_cfg() {
  local out_cfg="$1"
  local split_path="$2"
  local epoch_max="$3"
  local sr_ckpt="$4"
  local ocr_ckpt="$5"
  "${PY_RUN[@]}" - <<PY
import yaml
from pathlib import Path

template = Path("${PROJECT_DIR}") / "${CONFIG_TEMPLATE}"
runtime = Path("${out_cfg}")
cfg = yaml.safe_load(template.read_text(encoding="utf-8"))

cfg["MODEL_OCR"]["OCR_TRAIN"] = False
cfg["epoch_max"] = int("${epoch_max}")
cfg["resume"] = None
cfg["LOAD_PRE_TRAINED_SR"] = "${sr_ckpt}"
cfg["LOAD_PRE_TRAINED_OCR"] = "${ocr_ckpt}"

cfg["train_dataset"]["dataset"]["args"]["path_split"] = "${split_path}"
cfg["train_dataset"]["batch"] = int("${BATCH_SIZE}")
cfg["train_dataset"]["num_workers"] = int("${NUM_WORKERS}")
cfg["val_dataset"]["dataset"]["args"]["path_split"] = "${split_path}"
cfg["val_dataset"]["batch"] = int("${BATCH_SIZE}")
cfg["val_dataset"]["num_workers"] = int("${NUM_WORKERS}")

runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[INFO] runtime config saved: {runtime}")
PY
}

find_best_ckpt_epoch() {
  local dir="$1"
  local prefix="$2"
  local best_epoch=-1
  local best_file=""
  shopt -s nullglob
  local f base ep
  for f in "${dir}/${prefix}"_Epoch_*.pth; do
    base="$(basename "${f}")"
    ep="${base##*_Epoch_}"
    ep="${ep%.pth}"
    if [[ "${ep}" =~ ^[0-9]+$ ]]; then
      if (( ep > best_epoch )); then
        best_epoch="${ep}"
        best_file="${f}"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -z "${best_file}" ]]; then
    return 1
  fi
  echo "${best_file}"
}

echo "[INFO] start stage1 training"
make_runtime_cfg "${STAGE1_RUNTIME_CFG}" "${STAGE1_PREPROC_SPLIT}" "${STAGE1_EPOCH_MAX}" "${PRETRAINED_SR_CKPT}" "${PRETRAINED_OCR_CKPT}"
"${PY_RUN[@]}" ParallelNetTrain.py --config "${STAGE1_RUNTIME_CFG}" --save "${STAGE1_SAVE_TAG}"

if [[ ! -d "${STAGE1_SAVE_DIR}" ]]; then
  echo "[ERROR] stage1 save dir missing: ${STAGE1_SAVE_DIR}"
  exit 1
fi

STAGE1_BEST_SR="$(find_best_ckpt_epoch "${STAGE1_SAVE_DIR}" "best_model_cgnetV2_deformable" || true)"
STAGE1_BEST_OCR="$(find_best_ckpt_epoch "${STAGE1_SAVE_DIR}" "best_model_GPLPR" || true)"
if [[ -z "${STAGE1_BEST_SR}" || -z "${STAGE1_BEST_OCR}" ]]; then
  echo "[ERROR] failed to find stage1 best SR/OCR checkpoints."
  ls -la "${STAGE1_SAVE_DIR}" || true
  exit 1
fi
echo "[INFO] stage1 best SR: ${STAGE1_BEST_SR}"
echo "[INFO] stage1 best OCR: ${STAGE1_BEST_OCR}"

echo "[INFO] start stage2 training"
make_runtime_cfg "${STAGE2_RUNTIME_CFG}" "${STAGE2_PREPROC_SPLIT}" "${STAGE2_EPOCH_MAX}" "${STAGE1_BEST_SR}" "${STAGE1_BEST_OCR}"
"${PY_RUN[@]}" ParallelNetTrain.py --config "${STAGE2_RUNTIME_CFG}" --save "${STAGE2_SAVE_TAG}"

if [[ ! -d "${STAGE2_SAVE_DIR}" ]]; then
  echo "[ERROR] stage2 save dir missing: ${STAGE2_SAVE_DIR}"
  exit 1
fi

mkdir -p "${RUN_ROOT}/stage1" "${RUN_ROOT}/stage2" "${RUN_ROOT}/splits"
cp -a "${STAGE1_SAVE_DIR}/." "${RUN_ROOT}/stage1/"
cp -a "${STAGE2_SAVE_DIR}/." "${RUN_ROOT}/stage2/"

cp -f "${STAGE1_RUNTIME_CFG}" "${RUN_ROOT}/stage1_runtime.yaml"
cp -f "${STAGE2_RUNTIME_CFG}" "${RUN_ROOT}/stage2_runtime.yaml"
cp -f "${STAGE1_RAW_SPLIT}" "${RUN_ROOT}/splits/stage1_ab_split.txt"
cp -f "${STAGE2_RAW_SPLIT}" "${RUN_ROOT}/splits/stage2_b_split.txt"
cp -f "${STAGE1_PREPROC_SPLIT}" "${RUN_ROOT}/splits/stage1_ab_preprocessed.txt"
cp -f "${STAGE2_PREPROC_SPLIT}" "${RUN_ROOT}/splits/stage2_b_preprocessed.txt"

for f in "${SPLIT_OUT_ROOT}/track_selection.json" "${SPLIT_OUT_ROOT}/run_summary.json"; do
  if [[ -f "${f}" ]]; then
    cp -f "${f}" "${RUN_ROOT}/"
  fi
done

echo "[DONE] LCDNet 2-stage training finished."
echo "[DONE] run root: ${RUN_ROOT}"
echo "[DONE] stage1 save dir: ${STAGE1_SAVE_DIR}"
echo "[DONE] stage2 save dir: ${STAGE2_SAVE_DIR}"
