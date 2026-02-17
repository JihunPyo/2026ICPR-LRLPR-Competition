#!/bin/bash
set -euo pipefail

# One-shot pipeline:
#  1) Stage-1 re-adaptation from existing MF5 checkpoint (non-adversarial)
#  2) Stage-2 adversarial fine-tuning (GPLPR as discriminator)
#  3) Test-public SR inference
#  4) GPLPR OCR decode on generated SR
#
# Usage example:
#   PROJECT_DIR=/data/pjh7639/lpsr-lacd \
#   DATA_ROOT=/data/pjh7639/datasets/lpr_data \
#   BASE_MF5_CKPT=/data/pjh7639/datasets/mf5_runs/mf5_12345/best.pth \
#   OCR_CKPT=/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth \
#   bash scripts/slurm/train_mf5_adv_pipeline.sh

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

# Optional: W&B login
# Example:
# sbatch --export=ALL,WANDB_API_KEY=xxxxx,...
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] WANDB_API_KEY detected. Logging in to W&B."
  conda run --no-capture-output -n googlenet wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "[WARN] WANDB_API_KEY is not set. Use offline mode or provide key."
fi

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data}"
BASE_MF5_CKPT="${BASE_MF5_CKPT:-}"
OCR_CKPT="${OCR_CKPT:-/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth}"
RUN_TAG="${RUN_TAG:-adv_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/pjh7639/datasets/mf5_runs/${RUN_TAG}}"
INPUT_MODE="${INPUT_MODE:-stack15}"               # stack15 | center3
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE1_LCOFL_WEIGHT="${STAGE1_LCOFL_WEIGHT:-0.75}"
STAGE2_LCOFL_WEIGHT="${STAGE2_LCOFL_WEIGHT:-0.75}"
STAGE2_LCOFL_SCHED_ENABLED="${STAGE2_LCOFL_SCHED_ENABLED:-0}"  # 1: epoch-wise ramp
STAGE2_LCOFL_START_WEIGHT="${STAGE2_LCOFL_START_WEIGHT:-0.30}"
STAGE2_LCOFL_END_WEIGHT="${STAGE2_LCOFL_END_WEIGHT:-${STAGE2_LCOFL_WEIGHT}}"
STAGE2_LCOFL_START_EPOCH="${STAGE2_LCOFL_START_EPOCH:-1}"
STAGE2_LCOFL_END_EPOCH="${STAGE2_LCOFL_END_EPOCH:-${STAGE2_EPOCHS}}"
STAGE2_LCOFL_SCHED_MODE="${STAGE2_LCOFL_SCHED_MODE:-linear}"   # linear | cosine
STAGE2_CM_ENABLED="${STAGE2_CM_ENABLED:-1}"      # 1: update confusing pairs each epoch
STAGE2_CM_THRESHOLD="${STAGE2_CM_THRESHOLD:-0.25}"
STAGE2_CM_PAIR_WEIGHT="${STAGE2_CM_PAIR_WEIGHT:-0.5}"
ADV_G_WEIGHT="${ADV_G_WEIGHT:-0.2}"
ADV_D_STEPS="${ADV_D_STEPS:-1}"
ADV_WARMUP_EPOCHS="${ADV_WARMUP_EPOCHS:-2}"
ADV_REAL_FRAME_MODE="${ADV_REAL_FRAME_MODE:-random}"   # random | center | avg
ADV_D_LR="${ADV_D_LR:-1.0e-4}"
ADV_D_BETAS="${ADV_D_BETAS:-[0.9, 0.999]}"
ADV_D_LR_STEP="${ADV_D_LR_STEP:-10}"
ADV_D_LR_GAMMA="${ADV_D_LR_GAMMA:-0.5}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"  # 1: stage dataset to node-local SSD
NAS_DATA_ROOT="${NAS_DATA_ROOT:-/data/pjh7639/datasets}"
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-${NAS_DATA_ROOT}/raw-train.zip}"
RAW_TEST_ZIP="${RAW_TEST_ZIP:-${NAS_DATA_ROOT}/raw-test-public.zip}"
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"  # 1: keep local staged dataset after run

if [[ -z "${BASE_MF5_CKPT}" ]]; then
  echo "[ERROR] BASE_MF5_CKPT is required."
  exit 1
fi
if [[ ! -f "${BASE_MF5_CKPT}" ]]; then
  echo "[ERROR] BASE_MF5_CKPT not found: ${BASE_MF5_CKPT}"
  exit 1
fi
if [[ ! -f "${OCR_CKPT}" ]]; then
  echo "[ERROR] OCR_CKPT not found: ${OCR_CKPT}"
  exit 1
fi

mkdir -p "${RUN_ROOT}/configs"
cd "${PROJECT_DIR}"

STAGE1_CFG="${RUN_ROOT}/configs/mf5_stage1_readapt.yaml"
STAGE2_CFG="${RUN_ROOT}/configs/mf5_stage2_adv.yaml"
INFER_CFG="${RUN_ROOT}/configs/mf5_infer_test.yaml"
STAGE1_OUT="${RUN_ROOT}/stage1_readapt"
STAGE2_OUT="${RUN_ROOT}/stage2_adv"
INFER_OUT="${RUN_ROOT}/test_sr"
OCR_TXT="${RUN_ROOT}/test_sr_ocr_predictions.txt"

EFFECTIVE_DATA_ROOT="${DATA_ROOT}"
LOCAL_ROOT=""
LOCAL_DATA_DIR=""
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
  if [[ ! -f "${RAW_TRAIN_ZIP}" || ! -f "${RAW_TEST_ZIP}" ]]; then
    echo "[WARN] STAGE_TO_LOCAL=1 but raw zip files not found."
    echo "[WARN] RAW_TRAIN_ZIP=${RAW_TRAIN_ZIP}"
    echo "[WARN] RAW_TEST_ZIP=${RAW_TEST_ZIP}"
    echo "[WARN] fallback to DATA_ROOT=${DATA_ROOT}"
  else
    if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
      echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
      LOCAL_BASE="${HOME}/scratch/pjh7639"
      mkdir -p "${LOCAL_BASE}"
    fi

    LOCAL_ROOT="${LOCAL_BASE}/lpr_mf5_adv_${RUN_TAG}"
    LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
    LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
    mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}"

    echo "[INFO] staging dataset to local SSD: ${LOCAL_ROOT}"
    cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
    cp "${RAW_TEST_ZIP}" "${LOCAL_RAW_DIR}/"
    (
      cd "${LOCAL_RAW_DIR}"
      unzip -q -o "$(basename "${RAW_TRAIN_ZIP}")"
      unzip -q -o "$(basename "${RAW_TEST_ZIP}")"
    )

    TRAIN_PARENT="$(dirname "$(find "${LOCAL_RAW_DIR}" -type d -name 'Scenario-A' | head -n1)")"
    TEST_PARENT="$(dirname "$(dirname "$(find "${LOCAL_RAW_DIR}" -type f -name 'lr-001.jpg' | head -n1)")")"
    if [[ -z "${TRAIN_PARENT}" || -z "${TEST_PARENT}" ]]; then
      echo "[ERROR] failed to detect train/test roots after unzip on local SSD."
      exit 1
    fi
    ln -sfn "${TRAIN_PARENT}" "${LOCAL_DATA_DIR}/train"
    ln -sfn "${TEST_PARENT}" "${LOCAL_DATA_DIR}/test-public"
    EFFECTIVE_DATA_ROOT="${LOCAL_DATA_DIR}"
  fi
fi

if [[ ! -d "${EFFECTIVE_DATA_ROOT}/train" ]]; then
  echo "[ERROR] DATA_ROOT/train not found: ${EFFECTIVE_DATA_ROOT}/train"
  exit 1
fi
if [[ ! -d "${EFFECTIVE_DATA_ROOT}/test-public" ]]; then
  echo "[ERROR] DATA_ROOT/test-public not found: ${EFFECTIVE_DATA_ROOT}/test-public"
  exit 1
fi

echo "[INFO] project: ${PROJECT_DIR}"
echo "[INFO] data root: ${EFFECTIVE_DATA_ROOT}"
echo "[INFO] run root: ${RUN_ROOT}"
echo "[INFO] input mode: ${INPUT_MODE}"
echo "[INFO] base mf5 ckpt: ${BASE_MF5_CKPT}"
echo "[INFO] ocr ckpt: ${OCR_CKPT}"

${PY_RUN[@]} - <<PY
import yaml
from pathlib import Path

proj = Path("${PROJECT_DIR}")
stage1_cfg = Path("${STAGE1_CFG}")
stage2_cfg = Path("${STAGE2_CFG}")
infer_cfg = Path("${INFER_CFG}")

input_mode = "${INPUT_MODE}"
if input_mode not in {"stack15", "center3"}:
    raise ValueError(f"invalid INPUT_MODE: {input_mode}")

template = yaml.safe_load((proj / "mf5/configs/mf5_train_lcofl.yaml").read_text())
template["data"]["root"] = "${EFFECTIVE_DATA_ROOT}"
template["preprocess"]["input_mode"] = input_mode
if input_mode == "stack15":
    template["model"]["args"]["in_channels"] = 15
else:
    template["model"]["args"]["in_channels"] = 3
template["model"]["args"]["out_channels"] = 3
template["train"]["batch_size"] = int("${BATCH_SIZE}")
template["train"]["val_batch_size"] = int("${VAL_BATCH_SIZE}")
template["train"]["num_workers"] = int("${NUM_WORKERS}")
template["ocr"]["ocr_ckpt"] = "${OCR_CKPT}"

# Stage-1: non-adversarial re-adaptation
s1 = yaml.safe_load(yaml.safe_dump(template))
s1["model"]["checkpoint"] = "${BASE_MF5_CKPT}"
s1["train"]["resume"] = None
s1["train"]["epochs"] = int("${STAGE1_EPOCHS}")
s1["train"]["save_dir"] = "${STAGE1_OUT}"
s1["loss"]["lcofl_weight"] = float("${STAGE1_LCOFL_WEIGHT}")
s1["ocr"]["ocr_train"] = False
s1["adv"] = {
    "enabled": False,
    "g_weight": 0.0,
    "d_steps": 0,
    "warmup_epochs": 0,
    "real_frame_mode": "random",
}
stage1_cfg.write_text(yaml.safe_dump(s1, sort_keys=False))

# Stage-2: adversarial fine-tuning
s2 = yaml.safe_load(yaml.safe_dump(template))
s2["model"]["checkpoint"] = "${STAGE1_OUT}/best.pth"
s2["train"]["resume"] = None
s2["train"]["epochs"] = int("${STAGE2_EPOCHS}")
s2["train"]["save_dir"] = "${STAGE2_OUT}"
s2["loss"]["lcofl_weight"] = float("${STAGE2_LCOFL_WEIGHT}")
s2["loss"]["confusion_matrix"] = {
    "enabled": bool(int("${STAGE2_CM_ENABLED}")),
    "threshold": float("${STAGE2_CM_THRESHOLD}"),
    "pair_weight": float("${STAGE2_CM_PAIR_WEIGHT}"),
}
if int("${STAGE2_LCOFL_SCHED_ENABLED}") == 1:
    s2["loss"]["lcofl_schedule"] = {
        "enabled": True,
        "start_weight": float("${STAGE2_LCOFL_START_WEIGHT}"),
        "end_weight": float("${STAGE2_LCOFL_END_WEIGHT}"),
        "start_epoch": int("${STAGE2_LCOFL_START_EPOCH}"),
        "end_epoch": int("${STAGE2_LCOFL_END_EPOCH}"),
        "mode": "${STAGE2_LCOFL_SCHED_MODE}",
    }
s2["ocr"]["ocr_train"] = True
s2["adv"] = {
    "enabled": True,
    "g_weight": float("${ADV_G_WEIGHT}"),
    "d_steps": int("${ADV_D_STEPS}"),
    "warmup_epochs": int("${ADV_WARMUP_EPOCHS}"),
    "real_frame_mode": "${ADV_REAL_FRAME_MODE}",
    "d_lr": float("${ADV_D_LR}"),
    "d_betas": yaml.safe_load("${ADV_D_BETAS}"),
    "d_lr_step": int("${ADV_D_LR_STEP}"),
    "d_lr_gamma": float("${ADV_D_LR_GAMMA}"),
}
stage2_cfg.write_text(yaml.safe_dump(s2, sort_keys=False))

infer_template_name = "mf5_infer.yaml" if input_mode == "stack15" else "mf5_infer_center3.yaml"
infer_cfg_data = yaml.safe_load((proj / f"mf5/configs/{infer_template_name}").read_text())
infer_cfg_data["data"]["root"] = "${EFFECTIVE_DATA_ROOT}"
infer_cfg_data["preprocess"]["input_mode"] = input_mode
infer_cfg_data["infer"]["save_dir"] = "${INFER_OUT}"
infer_cfg.write_text(yaml.safe_dump(infer_cfg_data, sort_keys=False))

print(f"wrote {stage1_cfg}")
print(f"wrote {stage2_cfg}")
print(f"wrote {infer_cfg}")
PY

echo "[STAGE-1] Re-adaptation training"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE1_CFG}"

if [[ ! -f "${STAGE1_OUT}/best.pth" ]]; then
  echo "[ERROR] stage1 best checkpoint not found: ${STAGE1_OUT}/best.pth"
  exit 1
fi

echo "[STAGE-2] Adversarial fine-tuning"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE2_CFG}"

if [[ ! -f "${STAGE2_OUT}/best.pth" ]]; then
  echo "[ERROR] stage2 best checkpoint not found: ${STAGE2_OUT}/best.pth"
  exit 1
fi

echo "[INFER] Test-public SR generation"
${PY_RUN[@]} -m mf5.infer_mf5 --config "${INFER_CFG}" --checkpoint "${STAGE2_OUT}/best.pth"

echo "[OCR] GPLPR decode on generated SR"
${PY_RUN[@]} -m mf5.predict_test_ocr_gplpr \
  --sr-dir "${INFER_OUT}" \
  --ocr-checkpoint "${OCR_CKPT}" \
  --output-txt "${OCR_TXT}" \
  --batch-size 64 \
  --num-workers "${NUM_WORKERS}"

echo "[DONE] stage1: ${STAGE1_OUT}"
echo "[DONE] stage2: ${STAGE2_OUT}"
echo "[DONE] sr test dir: ${INFER_OUT}"
echo "[DONE] ocr txt: ${OCR_TXT}"
