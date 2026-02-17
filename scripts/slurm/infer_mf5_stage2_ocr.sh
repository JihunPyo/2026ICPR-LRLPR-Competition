#!/bin/bash
set -euo pipefail

# One-shot pipeline:
#   1) SR inference from Stage-2 checkpoint
#   2) OCR decoding from generated SR images
# Optional:
#   - Stage test-public dataset to node-local SSD and use it for inference.
#
# Usage example (recommended):
#   PROJECT_DIR=/data/pjh7639/lpsr-lacd \
#   RUN_ROOT=/data/pjh7639/datasets/mf5_runs/adv_20260216_043933 \
#   OCR_CKPT=/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth \
#   bash scripts/slurm/infer_mf5_stage2_ocr.sh
#
# Optional overrides:
#   STAGE2_CKPT=/data/.../stage2_adv/best.pth
#   INFER_CFG=/data/.../configs/mf5_infer_test.yaml
#   RUNTIME_INFER_CFG=/data/.../configs/mf5_infer_test.runtime.yaml
#   SR_DIR=/data/.../test_sr
#   NAS_SR_DIR=/data/.../test_sr
#   NAS_SR_ARCHIVE=/data/.../test_sr.tar.gz
#   OCR_TXT=/data/.../test_sr_ocr_predictions.txt
#   OCR_BATCH_SIZE=64 NUM_WORKERS=8
#   STAGE_TO_LOCAL=1 NAS_DATA_ROOT=/data/pjh7639/datasets

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
INFER_CFG="${INFER_CFG:-}"
RUNTIME_INFER_CFG="${RUNTIME_INFER_CFG:-}"
SR_DIR="${SR_DIR:-}"
NAS_SR_DIR="${NAS_SR_DIR:-}"
NAS_SR_ARCHIVE="${NAS_SR_ARCHIVE:-}"
OCR_CKPT="${OCR_CKPT:-/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth}"
OCR_TXT="${OCR_TXT:-}"
OCR_BATCH_SIZE="${OCR_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"
NAS_DATA_ROOT="${NAS_DATA_ROOT:-/data/pjh7639/datasets}"
RAW_TEST_ZIP="${RAW_TEST_ZIP:-${NAS_DATA_ROOT}/raw-test-public.zip}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"

if [[ -n "${RUN_ROOT}" ]]; then
  STAGE2_CKPT="${STAGE2_CKPT:-${RUN_ROOT}/stage2_adv/best.pth}"
  INFER_CFG="${INFER_CFG:-${RUN_ROOT}/configs/mf5_infer_test.yaml}"
  SR_DIR="${SR_DIR:-${RUN_ROOT}/test_sr}"
  NAS_SR_DIR="${NAS_SR_DIR:-${RUN_ROOT}/test_sr}"
  NAS_SR_ARCHIVE="${NAS_SR_ARCHIVE:-${RUN_ROOT}/test_sr.tar.gz}"
  OCR_TXT="${OCR_TXT:-${RUN_ROOT}/test_sr_ocr_predictions.txt}"
  RUNTIME_INFER_CFG="${RUNTIME_INFER_CFG:-${RUN_ROOT}/configs/mf5_infer_test.runtime.yaml}"
fi

if [[ -z "${STAGE2_CKPT}" ]]; then
  echo "[ERROR] STAGE2_CKPT is required (or set RUN_ROOT)."
  exit 1
fi
if [[ -z "${INFER_CFG}" ]]; then
  echo "[ERROR] INFER_CFG is required (or set RUN_ROOT)."
  exit 1
fi

if [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[ERROR] Stage-2 checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi
if [[ ! -f "${INFER_CFG}" ]]; then
  echo "[ERROR] Infer config not found: ${INFER_CFG}"
  exit 1
fi
if [[ ! -f "${OCR_CKPT}" ]]; then
  echo "[ERROR] OCR checkpoint not found: ${OCR_CKPT}"
  exit 1
fi

mapfile -t CFG_VALUES < <(
  ${PY_RUN[@]} - <<PY
import yaml
from pathlib import Path
cfg = Path("${INFER_CFG}")
data = yaml.safe_load(cfg.read_text())
print(str(data.get("data", {}).get("root", "")))
print(str(data.get("infer", {}).get("save_dir", "")))
PY
)
CFG_DATA_ROOT="${CFG_VALUES[0]:-}"
CFG_SR_DIR="${CFG_VALUES[1]:-}"

if [[ -z "${SR_DIR}" ]]; then
  SR_DIR="${CFG_SR_DIR}"
fi

if [[ -z "${SR_DIR}" ]]; then
  echo "[ERROR] SR_DIR is empty. Set SR_DIR or infer.save_dir in INFER_CFG."
  exit 1
fi

if [[ -z "${NAS_SR_DIR}" ]]; then
  NAS_SR_DIR="${SR_DIR}"
fi

if [[ -z "${NAS_SR_ARCHIVE}" ]]; then
  NAS_SR_ARCHIVE="${NAS_SR_DIR%/}.tar.gz"
fi

if [[ -z "${OCR_TXT}" ]]; then
  OCR_TXT="${NAS_SR_DIR%/}_ocr_predictions.txt"
fi

EFFECTIVE_DATA_ROOT="${CFG_DATA_ROOT}"
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
  if [[ ! -f "${RAW_TEST_ZIP}" ]]; then
    echo "[WARN] STAGE_TO_LOCAL=1 but RAW_TEST_ZIP not found: ${RAW_TEST_ZIP}"
    echo "[WARN] fallback to INFER_CFG data.root: ${CFG_DATA_ROOT}"
  else
    if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
      echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
      LOCAL_BASE="${HOME}/scratch/pjh7639"
      mkdir -p "${LOCAL_BASE}"
    fi

    RUN_TAG_SAFE="$(basename "${RUN_ROOT:-infer_$(date +%Y%m%d_%H%M%S)}" | tr -cs '[:alnum:]_-' '_')"
    LOCAL_ROOT="${LOCAL_BASE}/lpr_mf5_infer_${RUN_TAG_SAFE}"
    LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
    LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
    mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}"

    echo "[INFO] staging test dataset to local SSD: ${LOCAL_ROOT}"
    cp "${RAW_TEST_ZIP}" "${LOCAL_RAW_DIR}/"
    (
      cd "${LOCAL_RAW_DIR}"
      unzip -q -o "$(basename "${RAW_TEST_ZIP}")"
    )

    STAGED_TEST_DIR="${LOCAL_DATA_DIR}/test-public"
    rm -rf "${STAGED_TEST_DIR}"
    mkdir -p "${STAGED_TEST_DIR}"

    mapfile -t TEST_TRACK_DIRS < <(
      find "${LOCAL_RAW_DIR}" -type d \( -iname 'track_*' -o -iname 'track-*' \) | sort
    )
    if [[ "${#TEST_TRACK_DIRS[@]}" -eq 0 ]]; then
      mapfile -t TEST_TRACK_DIRS < <(
        find "${LOCAL_RAW_DIR}" -type f \( -name 'lr-001.jpg' -o -name 'lr-001.jpeg' -o -name 'lr-001.png' \) \
          -exec dirname {} \; | sort -u
      )
    fi

    if [[ "${#TEST_TRACK_DIRS[@]}" -eq 0 ]]; then
      echo "[ERROR] failed to detect any test track directories after unzip on local SSD."
      echo "[ERROR] unzip root: ${LOCAL_RAW_DIR}"
      find "${LOCAL_RAW_DIR}" -maxdepth 4 -type d | sed -n '1,80p'
      exit 1
    fi

    idx=0
    for tdir in "${TEST_TRACK_DIRS[@]}"; do
      base="$(basename "${tdir}")"
      name=""
      if [[ "${base}" =~ ^[Tt]rack_([0-9]+)$ ]]; then
        name="track_${BASH_REMATCH[1]}"
      elif [[ "${base}" =~ ^[Tt]rack-([0-9]+)$ ]]; then
        name="track_${BASH_REMATCH[1]}"
      fi
      if [[ -z "${name}" ]]; then
        name="$(printf "track_%06d" "${idx}")"
      fi
      while [[ -e "${STAGED_TEST_DIR}/${name}" ]]; do
        idx=$((idx + 1))
        name="$(printf "track_%06d" "${idx}")"
      done
      ln -sfn "${tdir}" "${STAGED_TEST_DIR}/${name}"
      idx=$((idx + 1))
    done

    EFFECTIVE_DATA_ROOT="${LOCAL_DATA_DIR}"

    TEST_TRACK_COUNT="$(find "${STAGED_TEST_DIR}" -mindepth 1 -maxdepth 1 -type l -name 'track_*' | wc -l | tr -d ' ')"
    echo "[INFO] staged test track count: ${TEST_TRACK_COUNT}"
    if [[ "${TEST_TRACK_COUNT}" == "0" ]]; then
      echo "[ERROR] no track_* symlinks built in staged test-public: ${STAGED_TEST_DIR}"
      find "${STAGED_TEST_DIR}" -maxdepth 2 -type d | sed -n '1,40p'
      exit 1
    fi

    # Normalize frame filenames for mf5 loader:
    # if lr-001 (no extension) exists, create lr-001.jpg symlink.
    NORMALIZED_BARE_COUNT=0
    for track_link in "${STAGED_TEST_DIR}"/track_*; do
      [[ -e "${track_link}" ]] || continue
      for i in 1 2 3 4 5; do
        stem="$(printf "lr-%03d" "${i}")"
        if find "${track_link}/" -maxdepth 1 -type f \
          \( -name "${stem}.jpg" -o -name "${stem}.jpeg" -o -name "${stem}.png" \
             -o -name "${stem}.JPG" -o -name "${stem}.JPEG" -o -name "${stem}.PNG" \) \
          | head -n1 | grep -q .; then
          continue
        fi
        if [[ -f "${track_link}/${stem}" ]]; then
          ln -sfn "${track_link}/${stem}" "${track_link}/${stem}.jpg"
          NORMALIZED_BARE_COUNT=$((NORMALIZED_BARE_COUNT + 1))
        fi
      done
    done
    if [[ "${NORMALIZED_BARE_COUNT}" != "0" ]]; then
      echo "[INFO] normalized bare lr-* filenames: ${NORMALIZED_BARE_COUNT}"
    fi

    VALID_TEST_TRACK_COUNT=0
    for track_link in "${STAGED_TEST_DIR}"/track_*; do
      [[ -e "${track_link}" ]] || continue
      valid=1
      for i in 1 2 3 4 5; do
        stem="$(printf "lr-%03d" "${i}")"
        if ! find "${track_link}/" -maxdepth 1 -type f \
          \( -name "${stem}.jpg" -o -name "${stem}.jpeg" -o -name "${stem}.png" \
             -o -name "${stem}.JPG" -o -name "${stem}.JPEG" -o -name "${stem}.PNG" \) \
          | head -n1 | grep -q .; then
          valid=0
          break
        fi
      done
      if [[ "${valid}" == "1" ]]; then
        VALID_TEST_TRACK_COUNT=$((VALID_TEST_TRACK_COUNT + 1))
      fi
    done
    echo "[INFO] staged valid test track count: ${VALID_TEST_TRACK_COUNT}"
    if [[ "${VALID_TEST_TRACK_COUNT}" == "0" ]]; then
      echo "[ERROR] staged tracks exist but none has complete lr-001..005 with supported extensions."
      echo "[ERROR] sample track listing:"
      SAMPLE_TRACK="$(find "${STAGED_TEST_DIR}" -mindepth 1 -maxdepth 1 -name 'track_*' | head -n1 || true)"
      if [[ -n "${SAMPLE_TRACK}" ]]; then
        ls -la "${SAMPLE_TRACK}/" | sed -n '1,80p'
      fi
      exit 1
    fi

    # If SR output target is the same as NAS target, write SR to local SSD first
    # and sync it back to NAS after inference.
    if [[ "${SR_DIR}" == "${NAS_SR_DIR}" ]]; then
      SR_DIR="${LOCAL_ROOT}/sr"
      echo "[INFO] using local SR runtime dir: ${SR_DIR}"
      echo "[INFO] SR will be synced to NAS dir: ${NAS_SR_DIR}"
    fi
  fi
fi

if [[ ! -d "${EFFECTIVE_DATA_ROOT}/test-public" ]]; then
  if [[ -n "${CFG_DATA_ROOT}" && -d "${CFG_DATA_ROOT}/test-public" ]]; then
    EFFECTIVE_DATA_ROOT="${CFG_DATA_ROOT}"
  else
    echo "[ERROR] test-public directory not found."
    echo "[ERROR] checked: ${EFFECTIVE_DATA_ROOT}/test-public"
    echo "[ERROR] cfg data.root: ${CFG_DATA_ROOT}"
    exit 1
  fi
fi

EFFECTIVE_TRACK_COUNT="$(find "${EFFECTIVE_DATA_ROOT}/test-public" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) -name 'track_*' | wc -l | tr -d ' ')"
echo "[INFO] effective test track count: ${EFFECTIVE_TRACK_COUNT}"
if [[ "${EFFECTIVE_TRACK_COUNT}" == "0" ]]; then
  echo "[ERROR] no track_* found in effective test-public: ${EFFECTIVE_DATA_ROOT}/test-public"
  find "${EFFECTIVE_DATA_ROOT}/test-public" -maxdepth 2 \( -type d -o -type l \) | sed -n '1,40p'
  exit 1
fi

if [[ -z "${RUNTIME_INFER_CFG}" ]]; then
  RUNTIME_INFER_CFG="/tmp/mf5_infer_test.runtime.$(date +%Y%m%d_%H%M%S).yaml"
fi
mkdir -p "$(dirname "${RUNTIME_INFER_CFG}")"

${PY_RUN[@]} - <<PY
import yaml
from pathlib import Path

src = Path("${INFER_CFG}")
dst = Path("${RUNTIME_INFER_CFG}")
cfg = yaml.safe_load(src.read_text())
cfg["data"]["root"] = "${EFFECTIVE_DATA_ROOT}"
cfg.setdefault("infer", {})
cfg["infer"]["save_dir"] = "${SR_DIR}"
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"saved runtime infer config: {dst}")
PY

cd "${PROJECT_DIR}"

mapfile -t DATASET_CHECK < <(
  ${PY_RUN[@]} - <<PY
from mf5.data import TrackSequenceDataset

root = "${EFFECTIVE_DATA_ROOT}"
ds = TrackSequenceDataset(data_root=root, phase="testing")
print(len(ds))
if len(ds) > 0:
    s = ds[0]
    print(s.get("track_id", ""))
    lr_paths = s.get("lr_paths", [])
    print("|".join(lr_paths))
PY
)
TEST_SAMPLE_COUNT="${DATASET_CHECK[0]:-0}"
echo "[INFO] testing dataset sample count: ${TEST_SAMPLE_COUNT}"
if [[ "${TEST_SAMPLE_COUNT}" == "0" ]]; then
  echo "[ERROR] TrackSequenceDataset(phase=testing) returned 0 samples."
  echo "[ERROR] effective data root: ${EFFECTIVE_DATA_ROOT}"
  echo "[ERROR] test-public listing:"
  ls -la "${EFFECTIVE_DATA_ROOT}/test-public" | sed -n '1,80p'
  exit 1
fi
if [[ "${#DATASET_CHECK[@]}" -ge 3 ]]; then
  echo "[INFO] first track id: ${DATASET_CHECK[1]}"
  echo "[INFO] first track lr paths: ${DATASET_CHECK[2]}"
fi

echo "[INFO] project: ${PROJECT_DIR}"
echo "[INFO] stage2 ckpt: ${STAGE2_CKPT}"
echo "[INFO] infer cfg (src): ${INFER_CFG}"
echo "[INFO] infer cfg (runtime): ${RUNTIME_INFER_CFG}"
echo "[INFO] data root (effective): ${EFFECTIVE_DATA_ROOT}"
echo "[INFO] sr dir: ${SR_DIR}"
echo "[INFO] nas sr dir: ${NAS_SR_DIR}"
echo "[INFO] nas sr archive: ${NAS_SR_ARCHIVE}"
echo "[INFO] ocr ckpt: ${OCR_CKPT}"
echo "[INFO] ocr txt: ${OCR_TXT}"

echo "[INFER] Stage-2 SR generation"
${PY_RUN[@]} -m mf5.infer_mf5 \
  --config "${RUNTIME_INFER_CFG}" \
  --checkpoint "${STAGE2_CKPT}"

if [[ ! -d "${SR_DIR}" ]]; then
  echo "[ERROR] SR output directory not found: ${SR_DIR}"
  exit 1
fi

if ! find "${SR_DIR}" -mindepth 2 -maxdepth 2 -type f -name "sr.png" -print -quit | grep -q .; then
  echo "[ERROR] No sr.png files found under ${SR_DIR}"
  exit 1
fi

OCR_SR_DIR="${SR_DIR}"

if [[ ! -d "${OCR_SR_DIR}" ]]; then
  echo "[ERROR] OCR SR directory not found: ${OCR_SR_DIR}"
  exit 1
fi
if ! find "${OCR_SR_DIR}" -mindepth 2 -maxdepth 2 -type f -name "sr.png" -print -quit | grep -q .; then
  echo "[ERROR] No sr.png files found under OCR SR directory: ${OCR_SR_DIR}"
  exit 1
fi

echo "[OCR] GPLPR decode from generated SR"
${PY_RUN[@]} -m mf5.predict_test_ocr_gplpr \
  --sr-dir "${OCR_SR_DIR}" \
  --ocr-checkpoint "${OCR_CKPT}" \
  --output-txt "${OCR_TXT}" \
  --batch-size "${OCR_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

SR_DIR_NORM="${SR_DIR%/}"
SR_PARENT="$(dirname "${SR_DIR_NORM}")"
SR_BASENAME="$(basename "${SR_DIR_NORM}")"
ARCHIVE_BASENAME="$(basename "${NAS_SR_ARCHIVE}")"
if [[ -n "${LOCAL_ROOT}" ]]; then
  LOCAL_SR_ARCHIVE="${LOCAL_ROOT}/${ARCHIVE_BASENAME}"
else
  LOCAL_SR_ARCHIVE="/tmp/${ARCHIVE_BASENAME}"
fi

echo "[ARCHIVE] pack SR images: ${LOCAL_SR_ARCHIVE}"
tar -C "${SR_PARENT}" -czf "${LOCAL_SR_ARCHIVE}" "${SR_BASENAME}"

echo "[SYNC] copy SR archive to NAS: ${NAS_SR_ARCHIVE}"
mkdir -p "$(dirname "${NAS_SR_ARCHIVE}")"
if [[ "${LOCAL_SR_ARCHIVE}" != "${NAS_SR_ARCHIVE}" ]]; then
  cp "${LOCAL_SR_ARCHIVE}" "${NAS_SR_ARCHIVE}"
fi

echo "[DONE] sr dir (runtime): ${SR_DIR}"
echo "[DONE] sr archive (nas): ${NAS_SR_ARCHIVE}"
echo "[DONE] ocr txt: ${OCR_TXT}"
