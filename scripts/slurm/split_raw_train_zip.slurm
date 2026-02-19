#!/bin/bash

#SBATCH -J split_raw_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -p batch_ugrad
#SBATCH -t 0-08:00:00
#SBATCH -o slurm-%A.split_raw_train.out

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
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-/data/pjh7639/datasets/raw-train.zip}"
RAW_TRAIN_VAL_ZIP="${RAW_TRAIN_VAL_ZIP:-/data/pjh7639/datasets/raw-train-trainval.zip}"
RAW_TRAIN_TEST_ZIP="${RAW_TRAIN_TEST_ZIP:-/data/pjh7639/datasets/raw-train-test.zip}"
TEST_RATIO="${TEST_RATIO:-0.1}"
TEST_SCENARIO="${TEST_SCENARIO:-B}"
SEED="${SEED:-1996}"
OVERWRITE="${OVERWRITE:-1}"
STRICT="${STRICT:-1}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

# Metadata outputs are written next to val zip by default.
MANIFEST_ZIP_DIR_DEFAULT="$(dirname "${RAW_TRAIN_VAL_ZIP}")"
MANIFEST_ZIP_DIR="${MANIFEST_ZIP_DIR:-${MANIFEST_ZIP_DIR_DEFAULT}}"
MANIFEST_JSON="${MANIFEST_JSON:-${MANIFEST_ZIP_DIR}/raw-train_split_manifest.json}"
VAL_TRACK_LIST="${VAL_TRACK_LIST:-${MANIFEST_ZIP_DIR}/raw-train_val_tracks.txt}"
TEST_TRACK_LIST="${TEST_TRACK_LIST:-${MANIFEST_ZIP_DIR}/raw-train_test_tracks.txt}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi

SPLIT_SCRIPT="${PROJECT_DIR}/scripts/preprocess/split_raw_train_zip.py"
if [[ ! -f "${SPLIT_SCRIPT}" ]]; then
  echo "[ERROR] split script not found: ${SPLIT_SCRIPT}"
  exit 1
fi

if [[ ! -f "${RAW_TRAIN_ZIP}" ]]; then
  echo "[ERROR] RAW_TRAIN_ZIP not found: ${RAW_TRAIN_ZIP}"
  exit 1
fi

if [[ "${RAW_TRAIN_VAL_ZIP}" == "${RAW_TRAIN_TEST_ZIP}" ]]; then
  echo "[ERROR] RAW_TRAIN_VAL_ZIP and RAW_TRAIN_TEST_ZIP must be different."
  exit 1
fi

RUN_TAG_DEFAULT="split_raw_train_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"

EFFECTIVE_INPUT_ZIP="${RAW_TRAIN_ZIP}"
LOCAL_ROOT=""
LOCAL_RAW_DIR=""
LOCAL_OUT_DIR=""
LOCAL_MANIFEST=""
LOCAL_VAL_LIST=""
LOCAL_TEST_LIST=""
LOCAL_VAL_ZIP=""
LOCAL_TEST_ZIP=""

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

  LOCAL_ROOT="${LOCAL_BASE}/split_raw_train_${RUN_TAG}"
  LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
  LOCAL_OUT_DIR="${LOCAL_ROOT}/out"
  mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_OUT_DIR}"

  echo "[INFO] staging input zip to local SSD: ${LOCAL_ROOT}"
  cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
  EFFECTIVE_INPUT_ZIP="${LOCAL_RAW_DIR}/$(basename "${RAW_TRAIN_ZIP}")"

  LOCAL_VAL_ZIP="${LOCAL_OUT_DIR}/raw-train-trainval.zip"
  LOCAL_TEST_ZIP="${LOCAL_OUT_DIR}/raw-train-test.zip"
  LOCAL_MANIFEST="${LOCAL_OUT_DIR}/raw-train_split_manifest.json"
  LOCAL_VAL_LIST="${LOCAL_OUT_DIR}/raw-train_val_tracks.txt"
  LOCAL_TEST_LIST="${LOCAL_OUT_DIR}/raw-train_test_tracks.txt"
else
  LOCAL_VAL_ZIP="${RAW_TRAIN_VAL_ZIP}"
  LOCAL_TEST_ZIP="${RAW_TRAIN_TEST_ZIP}"
  LOCAL_MANIFEST="${MANIFEST_JSON}"
  LOCAL_VAL_LIST="${VAL_TRACK_LIST}"
  LOCAL_TEST_LIST="${TEST_TRACK_LIST}"
fi

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] input zip: ${EFFECTIVE_INPUT_ZIP}"
echo "[INFO] out val zip: ${RAW_TRAIN_VAL_ZIP}"
echo "[INFO] out test zip: ${RAW_TRAIN_TEST_ZIP}"
echo "[INFO] test ratio: ${TEST_RATIO}"
echo "[INFO] test scenario: ${TEST_SCENARIO}"
echo "[INFO] seed: ${SEED}"
echo "[INFO] overwrite: ${OVERWRITE}"
echo "[INFO] strict: ${STRICT}"
echo "[INFO] local staging: ${STAGE_TO_LOCAL}"
echo "[INFO] note: GPU is allocated by request, but this is CPU-bound I/O work."

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

"${PY_RUN[@]}" "${SPLIT_SCRIPT}" \
  --input-zip "${EFFECTIVE_INPUT_ZIP}" \
  --out-val-zip "${LOCAL_VAL_ZIP}" \
  --out-test-zip "${LOCAL_TEST_ZIP}" \
  --test-ratio "${TEST_RATIO}" \
  --test-scenario "${TEST_SCENARIO}" \
  --seed "${SEED}" \
  --manifest-json "${LOCAL_MANIFEST}" \
  --val-track-list "${LOCAL_VAL_LIST}" \
  --test-track-list "${LOCAL_TEST_LIST}" \
  --overwrite "${OVERWRITE}" \
  --strict "${STRICT}"

# Verify output zip root layout.
verify_zip_root() {
  local z="$1"
  if ! "${PY_RUN[@]}" - "$z" <<'PY'
import sys
import zipfile

zip_path = sys.argv[1]
bad = []
with zipfile.ZipFile(zip_path, "r") as zf:
    for name in zf.namelist():
        if name and not name.startswith("train/"):
            bad.append(name)

if bad:
    print(f"[ERROR] zip contains entries not under train/: {zip_path}")
    print("[ERROR] sample bad entries:")
    for name in bad[:20]:
        print(f"  - {name}")
    raise SystemExit(1)
PY
  then
    exit 1
  fi
  echo "[INFO] verified zip root train/: ${z}"
}

for z in "${LOCAL_VAL_ZIP}" "${LOCAL_TEST_ZIP}"; do
  verify_zip_root "${z}"
done

atomic_publish_file() {
  local src="$1"
  local dst="$2"
  local dstdir tmp
  dstdir="$(dirname "${dst}")"
  mkdir -p "${dstdir}"
  tmp="${dst}.tmp.${SLURM_JOB_ID:-manual}.$RANDOM"
  cp "${src}" "${tmp}"
  mv -f "${tmp}" "${dst}"
}

if [[ "${STAGE_TO_LOCAL}" == "1" ]]; then
  echo "[INFO] publishing artifacts to NAS (atomic rename)..."
  atomic_publish_file "${LOCAL_VAL_ZIP}" "${RAW_TRAIN_VAL_ZIP}"
  atomic_publish_file "${LOCAL_TEST_ZIP}" "${RAW_TRAIN_TEST_ZIP}"
  atomic_publish_file "${LOCAL_MANIFEST}" "${MANIFEST_JSON}"
  atomic_publish_file "${LOCAL_VAL_LIST}" "${VAL_TRACK_LIST}"
  atomic_publish_file "${LOCAL_TEST_LIST}" "${TEST_TRACK_LIST}"
fi

echo "[DONE] split artifacts"
echo "[DONE] val zip: ${RAW_TRAIN_VAL_ZIP}"
echo "[DONE] test zip: ${RAW_TRAIN_TEST_ZIP}"
echo "[DONE] manifest: ${MANIFEST_JSON}"
echo "[DONE] val tracks: ${VAL_TRACK_LIST}"
echo "[DONE] test tracks: ${TEST_TRACK_LIST}"
