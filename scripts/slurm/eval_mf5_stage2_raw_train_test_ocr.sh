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
RUN_ROOT="${RUN_ROOT:-}"
STAGE2_CKPT="${STAGE2_CKPT:-}"
TRAIN_CFG="${TRAIN_CFG:-}"
OCR_CKPT="${OCR_CKPT:-}"

RAW_TEST_ZIP="${RAW_TEST_ZIP:-/data/pjh7639/datasets/raw-train-test.zip}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"     # 1: stage zip to node-local SSD
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"   # 1: keep staged local files
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"
DATA_ROOT="${DATA_ROOT:-}"                # required when STAGE_TO_LOCAL=0 and must contain train/

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/pjh7639/weights/mf5/evals}"
RUN_TAG_DEFAULT="mf5_s2_raw_train_test_ocr_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
RUNTIME_CFG="${RUNTIME_CFG:-${OUT_DIR}/mf5_stage2_raw_train_test.runtime.yaml}"
SR_SAVE_DIR="${SR_SAVE_DIR:-}"            # empty -> auto resolve (local SSD if enabled, else OUT_DIR/sr)
NAS_SR_ZIP="${NAS_SR_ZIP:-/data/pjh7639/datasets/mf5-sr/sr-test-mf5_v2.zip}"
ZIP_COMPRESS_LEVEL="${ZIP_COMPRESS_LEVEL:-6}"

VAL_RATIO="${VAL_RATIO:-1.0}"             # 1.0 => use all tracks as validation in eval_val_ocr script
BATCH_SIZE="${BATCH_SIZE:-}"              # empty => use train.val_batch_size from train cfg
NUM_WORKERS="${NUM_WORKERS:-8}"

OCR_ALPHABET="${OCR_ALPHABET:-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ}"
OCR_NC="${OCR_NC:-3}"
OCR_K="${OCR_K:-7}"
OCR_IS_SEQ_MODEL="${OCR_IS_SEQ_MODEL:-1}"
OCR_HEAD="${OCR_HEAD:-2}"
OCR_INNER="${OCR_INNER:-256}"
OCR_IS_L2_NORM="${OCR_IS_L2_NORM:-1}"

if [[ -n "${RUN_ROOT}" ]]; then
  STAGE2_CKPT="${STAGE2_CKPT:-${RUN_ROOT}/stage2_adv/best.pth}"
  TRAIN_CFG="${TRAIN_CFG:-${RUN_ROOT}/configs/mf5_stage2_adv.yaml}"
fi

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ -z "${STAGE2_CKPT}" || ! -f "${STAGE2_CKPT}" ]]; then
  echo "[ERROR] STAGE2_CKPT not found: ${STAGE2_CKPT:-<unset>}"
  exit 1
fi
if [[ -z "${TRAIN_CFG}" || ! -f "${TRAIN_CFG}" ]]; then
  echo "[ERROR] TRAIN_CFG not found: ${TRAIN_CFG:-<unset>}"
  exit 1
fi

if [[ ! -f "${RAW_TEST_ZIP}" ]]; then
  RAW_TEST_ZIP_FIXED="${RAW_TEST_ZIP/raw-trian-test/raw-train-test}"
  if [[ "${RAW_TEST_ZIP_FIXED}" != "${RAW_TEST_ZIP}" && -f "${RAW_TEST_ZIP_FIXED}" ]]; then
    echo "[WARN] RAW_TEST_ZIP typo detected. Using: ${RAW_TEST_ZIP_FIXED}"
    RAW_TEST_ZIP="${RAW_TEST_ZIP_FIXED}"
  fi
fi

has_scenario_dirs() {
  local base="$1"
  local any_dir
  any_dir="$(find "${base}" -maxdepth 1 -mindepth 1 -type d \( -iname 'Scenario-*' -o -iname 'Senario-*' \) | head -n1 || true)"
  [[ -n "${any_dir}" ]]
}

resolve_train_root() {
  local base="$1"
  if [[ ! -d "${base}" ]]; then
    return 1
  fi
  if has_scenario_dirs "${base}"; then
    echo "${base}"
    return 0
  fi
  if [[ -d "${base}/train" ]] && has_scenario_dirs "${base}/train"; then
    echo "${base}/train"
    return 0
  fi
  mapfile -t parents < <(
    find "${base}" -type d \( -iname 'Scenario-*' -o -iname 'Senario-*' \) \
      -exec dirname {} \; | sort -u
  )
  local p
  for p in "${parents[@]}"; do
    if has_scenario_dirs "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

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

EFFECTIVE_DATA_ROOT=""
if [[ "${STAGE_TO_LOCAL}" == "1" ]]; then
  if [[ ! -f "${RAW_TEST_ZIP}" ]]; then
    echo "[ERROR] RAW_TEST_ZIP not found: ${RAW_TEST_ZIP}"
    exit 1
  fi

  if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
    echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
    LOCAL_BASE="${HOME}/scratch/pjh7639"
    mkdir -p "${LOCAL_BASE}"
  fi

  LOCAL_ROOT="${LOCAL_BASE}/mf5_s2_raw_train_test_ocr_${RUN_TAG}"
  LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
  LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
  mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}"

  echo "[INFO] staging raw-train-test zip to local SSD: ${LOCAL_ROOT}"
  cp "${RAW_TEST_ZIP}" "${LOCAL_RAW_DIR}/"
  (
    cd "${LOCAL_RAW_DIR}"
    unzip -q -o "$(basename "${RAW_TEST_ZIP}")"
  )

  TRAIN_ROOT="$(resolve_train_root "${LOCAL_RAW_DIR}" || true)"
  if [[ -z "${TRAIN_ROOT}" ]]; then
    echo "[ERROR] failed to resolve train root from staged zip."
    find "${LOCAL_RAW_DIR}" -maxdepth 4 -type d | sed -n '1,120p' || true
    exit 1
  fi

  ln -sfn "${TRAIN_ROOT}" "${LOCAL_DATA_DIR}/train"
  EFFECTIVE_DATA_ROOT="${LOCAL_DATA_DIR}"
else
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "[ERROR] set DATA_ROOT when STAGE_TO_LOCAL=0. DATA_ROOT must contain train/."
    exit 1
  fi
  if [[ ! -d "${DATA_ROOT}/train" ]]; then
    echo "[ERROR] DATA_ROOT/train not found: ${DATA_ROOT}/train"
    exit 1
  fi
  EFFECTIVE_DATA_ROOT="${DATA_ROOT}"
fi

mkdir -p "${OUT_DIR}"
mkdir -p "$(dirname "${RUNTIME_CFG}")"
if [[ -z "${SR_SAVE_DIR}" ]]; then
  if [[ -n "${LOCAL_ROOT}" ]]; then
    SR_SAVE_DIR="${LOCAL_ROOT}/sr"
  else
    SR_SAVE_DIR="${OUT_DIR}/sr"
  fi
fi
mkdir -p "${SR_SAVE_DIR}"

mapfile -t CFG_INFO < <(
  ${PY_RUN[@]} - <<PY
import yaml
from pathlib import Path

src = Path("${TRAIN_CFG}")
dst = Path("${RUNTIME_CFG}")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.setdefault("data", {})
cfg["data"]["root"] = "${EFFECTIVE_DATA_ROOT}"
cfg["data"]["val_ratio"] = float("${VAL_RATIO}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(cfg.get("ocr", {}).get("ocr_ckpt", "")))
print(str(cfg.get("train", {}).get("val_batch_size", 32)))
print(str(dst))
PY
)

CFG_OCR_CKPT="${CFG_INFO[0]:-}"
CFG_VAL_BATCH_SIZE="${CFG_INFO[1]:-32}"
RUNTIME_CFG="${CFG_INFO[2]:-${RUNTIME_CFG}}"

if [[ -z "${OCR_CKPT}" ]]; then
  OCR_CKPT="${CFG_OCR_CKPT}"
fi
if [[ -z "${OCR_CKPT}" ]]; then
  echo "[ERROR] OCR_CKPT is empty. Set OCR_CKPT explicitly or include ocr.ocr_ckpt in TRAIN_CFG."
  exit 1
fi
if [[ ! -f "${OCR_CKPT}" ]]; then
  echo "[ERROR] OCR_CKPT not found: ${OCR_CKPT}"
  exit 1
fi

if [[ -z "${BATCH_SIZE}" ]]; then
  BATCH_SIZE="${CFG_VAL_BATCH_SIZE}"
fi

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] run root: ${RUN_ROOT:-<unset>}"
echo "[INFO] stage2 checkpoint: ${STAGE2_CKPT}"
echo "[INFO] train cfg: ${TRAIN_CFG}"
echo "[INFO] runtime train cfg: ${RUNTIME_CFG}"
echo "[INFO] raw test zip: ${RAW_TEST_ZIP}"
echo "[INFO] data root (effective): ${EFFECTIVE_DATA_ROOT}"
echo "[INFO] output dir: ${OUT_DIR}"
echo "[INFO] sr save dir: ${SR_SAVE_DIR}"
echo "[INFO] nas sr zip: ${NAS_SR_ZIP}"
echo "[INFO] val_ratio: ${VAL_RATIO}"
echo "[INFO] batch_size: ${BATCH_SIZE}"
echo "[INFO] num_workers: ${NUM_WORKERS}"
echo "[INFO] ocr checkpoint: ${OCR_CKPT}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

${PY_RUN[@]} -m mf5.eval_val_ocr_gplpr_20260223 \
  --train-config "${RUNTIME_CFG}" \
  --sr-checkpoint "${STAGE2_CKPT}" \
  --ocr-checkpoint "${OCR_CKPT}" \
  --output-dir "${OUT_DIR}" \
  --save-sr-dir "${SR_SAVE_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --ocr-alphabet "${OCR_ALPHABET}" \
  --ocr-nc "${OCR_NC}" \
  --ocr-k "${OCR_K}" \
  --ocr-is-seq-model "${OCR_IS_SEQ_MODEL}" \
  --ocr-head "${OCR_HEAD}" \
  --ocr-inner "${OCR_INNER}" \
  --ocr-is-l2-norm "${OCR_IS_L2_NORM}"

if ! find "${SR_SAVE_DIR}" -mindepth 2 -maxdepth 2 -type f -name "sr.png" -print -quit | grep -q .; then
  echo "[ERROR] no sr.png files found under SR_SAVE_DIR: ${SR_SAVE_DIR}"
  exit 1
fi

SR_ZIP_BASENAME="$(basename "${NAS_SR_ZIP}")"
if [[ -n "${LOCAL_ROOT}" ]]; then
  LOCAL_SR_ZIP="${LOCAL_ROOT}/${SR_ZIP_BASENAME}"
else
  LOCAL_SR_ZIP="${OUT_DIR}/${SR_ZIP_BASENAME}"
fi

${PY_RUN[@]} - <<PY
import zipfile
from pathlib import Path

sr_dir = Path("${SR_SAVE_DIR}")
zip_path = Path("${LOCAL_SR_ZIP}")
zip_path.parent.mkdir(parents=True, exist_ok=True)

files = sorted([p for p in sr_dir.rglob("*") if p.is_file()])
if not files:
    raise RuntimeError(f"no files found under SR save dir: {sr_dir}")

with zipfile.ZipFile(
    zip_path,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=int("${ZIP_COMPRESS_LEVEL}"),
    allowZip64=True,
) as zf:
    for p in files:
        zf.write(p, arcname=str(p.relative_to(sr_dir)))

print(f"[INFO] local sr zip: {zip_path}")
print(f"[INFO] file_count: {len(files)}")
PY

mkdir -p "$(dirname "${NAS_SR_ZIP}")"
if [[ "${LOCAL_SR_ZIP}" != "${NAS_SR_ZIP}" ]]; then
  cp "${LOCAL_SR_ZIP}" "${NAS_SR_ZIP}"
fi
echo "[DONE] sr zip (nas): ${NAS_SR_ZIP}"

SUMMARY_JSON="${OUT_DIR}/val_ocr_summary.json"
if [[ -f "${SUMMARY_JSON}" ]]; then
  mapfile -t SUMMARY_INFO < <(
    ${PY_RUN[@]} - <<PY
import json
from pathlib import Path
s = json.loads(Path("${SUMMARY_JSON}").read_text(encoding="utf-8"))
print(s.get("num_samples", 0))
print(s.get("sr_exact_match_acc", 0.0))
print(s.get("sr_cer", 0.0))
print(s.get("lr_center_exact_match_acc", 0.0))
print(s.get("lr_center_cer", 0.0))
PY
  )
  echo "[RESULT] num_samples: ${SUMMARY_INFO[0]}"
  echo "[RESULT] SR OCR exact-match acc: ${SUMMARY_INFO[1]}"
  echo "[RESULT] SR OCR CER: ${SUMMARY_INFO[2]}"
  echo "[RESULT] LR-center OCR exact-match acc: ${SUMMARY_INFO[3]}"
  echo "[RESULT] LR-center OCR CER: ${SUMMARY_INFO[4]}"
fi

echo "[DONE] evaluation finished"
echo "[DONE] per-sample csv: ${OUT_DIR}/val_ocr_per_sample.csv"
echo "[DONE] summary json: ${SUMMARY_JSON}"
