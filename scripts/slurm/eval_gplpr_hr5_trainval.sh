#!/bin/bash
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
  echo "[ERROR] conda binary not found in this environment."
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
TRAIN_ROOT_NAS="${TRAIN_ROOT_NAS:-/data/pjh7639/datasets/raw_train_unzip}"
OCR_CKPT="${OCR_CKPT:-/data/pjh7639/weights/GP_LPR/Rodosol.pth}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/pjh7639/weights/GP_LPR/evals}"
RUN_TAG_DEFAULT="gplpr_hr5_eval_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"

STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

STRICT_COMPLETE_HR5="${STRICT_COMPLETE_HR5:-1}"
MAX_TRACKS="${MAX_TRACKS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-auto}"
TEMP_ROOT="${TEMP_ROOT:-}"

OCR_ALPHABET="${OCR_ALPHABET:-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ}"
OCR_NC="${OCR_NC:-3}"
OCR_K="${OCR_K:-7}"
OCR_IS_SEQ_MODEL="${OCR_IS_SEQ_MODEL:-1}"
OCR_HEAD="${OCR_HEAD:-2}"
OCR_INNER="${OCR_INNER:-256}"
OCR_IS_L2_NORM="${OCR_IS_L2_NORM:-1}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${OCR_CKPT}" ]]; then
  echo "[ERROR] OCR_CKPT not found: ${OCR_CKPT}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

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

INPUT_MODE="train_root"
INPUT_VALUE="${TRAIN_ROOT_NAS}"

if [[ "${STAGE_TO_LOCAL}" == "1" ]]; then
  if [[ ! -f "${RAW_TRAIN_ZIP}" ]]; then
    echo "[WARN] STAGE_TO_LOCAL=1 but RAW_TRAIN_ZIP not found: ${RAW_TRAIN_ZIP}"
    echo "[WARN] fallback to TRAIN_ROOT_NAS or input-zip mode."
  else
    if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
      echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
      LOCAL_BASE="${HOME}/scratch/pjh7639"
      mkdir -p "${LOCAL_BASE}"
    fi

    LOCAL_ROOT="${LOCAL_BASE}/gplpr_hr5_eval_${RUN_TAG}"
    LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
    mkdir -p "${LOCAL_RAW_DIR}"

    echo "[INFO] staging raw-train-trainval.zip to local SSD: ${LOCAL_ROOT}"
    cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
    (
      cd "${LOCAL_RAW_DIR}"
      unzip -q -o "$(basename "${RAW_TRAIN_ZIP}")"
    )

    INPUT_MODE="train_root"
    INPUT_VALUE="${LOCAL_RAW_DIR}"
  fi
fi

if [[ "${INPUT_MODE}" == "train_root" && ! -d "${INPUT_VALUE}" ]]; then
  if [[ -f "${RAW_TRAIN_ZIP}" ]]; then
    INPUT_MODE="input_zip"
    INPUT_VALUE="${RAW_TRAIN_ZIP}"
  else
    echo "[ERROR] train root not found and raw zip not found."
    echo "[ERROR] TRAIN_ROOT_NAS=${TRAIN_ROOT_NAS}"
    echo "[ERROR] RAW_TRAIN_ZIP=${RAW_TRAIN_ZIP}"
    exit 1
  fi
fi

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] input mode: ${INPUT_MODE}"
echo "[INFO] input value: ${INPUT_VALUE}"
echo "[INFO] ocr ckpt: ${OCR_CKPT}"
echo "[INFO] stage_to_local: ${STAGE_TO_LOCAL}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

ARGS=(
  -m mf5.eval_trainval_hr5_ocr_gplpr
  --output-dir "${OUT_DIR}"
  --ocr-checkpoint "${OCR_CKPT}"
  --strict-complete-hr5 "${STRICT_COMPLETE_HR5}"
  --max-tracks "${MAX_TRACKS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --ocr-alphabet "${OCR_ALPHABET}"
  --ocr-nc "${OCR_NC}"
  --ocr-k "${OCR_K}"
  --ocr-is-seq-model "${OCR_IS_SEQ_MODEL}"
  --ocr-head "${OCR_HEAD}"
  --ocr-inner "${OCR_INNER}"
  --ocr-is-l2-norm "${OCR_IS_L2_NORM}"
)

if [[ -n "${TEMP_ROOT}" ]]; then
  ARGS+=(--temp-root "${TEMP_ROOT}")
fi

if [[ "${INPUT_MODE}" == "train_root" ]]; then
  ARGS+=(--train-root "${INPUT_VALUE}")
else
  ARGS+=(--input-zip "${INPUT_VALUE}")
fi

"${PY_RUN[@]}" "${ARGS[@]}"

echo "[DONE] evaluation finished"
echo "[DONE] out dir: ${OUT_DIR}"
echo "[DONE] summary: ${OUT_DIR}/run_summary.json"
