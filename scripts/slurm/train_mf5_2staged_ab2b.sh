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
  echo "[ERROR] conda binary not found in this batch environment."
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "googlenet"; then
  echo "[ERROR] conda env not found: googlenet"
  conda env list || true
  exit 1
fi

PY_RUN=(conda run --no-capture-output -n googlenet python -u)

# W&B auth strategy:
# - default: use WANDB_API_KEY env directly (non-blocking, recommended for batch jobs)
# - optional: explicit login attempt with timeout (best_effort / required)
WANDB_LOGIN_MODE="${WANDB_LOGIN_MODE:-skip}"          # skip | best_effort | required
WANDB_LOGIN_TIMEOUT_SEC="${WANDB_LOGIN_TIMEOUT_SEC:-20}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  echo "[INFO] WANDB_API_KEY detected. Using env-based W&B authentication."
  if [[ "${WANDB_LOGIN_MODE}" != "skip" ]]; then
    echo "[INFO] WANDB_LOGIN_MODE=${WANDB_LOGIN_MODE}. Trying explicit wandb login (timeout=${WANDB_LOGIN_TIMEOUT_SEC}s)."
    if command -v timeout >/dev/null 2>&1; then
      if timeout "${WANDB_LOGIN_TIMEOUT_SEC}" \
        conda run --no-capture-output -n googlenet wandb login --relogin "${WANDB_API_KEY}"; then
        echo "[INFO] wandb login completed."
      else
        rc=$?
        if [[ "${WANDB_LOGIN_MODE}" == "required" ]]; then
          echo "[ERROR] wandb login failed/timed out (rc=${rc}) and WANDB_LOGIN_MODE=required."
          exit 1
        fi
        echo "[WARN] wandb login failed/timed out (rc=${rc}); continue with env-based auth."
      fi
    else
      if [[ "${WANDB_LOGIN_MODE}" == "required" ]]; then
        echo "[ERROR] timeout command not found but WANDB_LOGIN_MODE=required."
        exit 1
      fi
      echo "[WARN] timeout command not found; skip explicit wandb login and continue."
    fi
  fi
else
  echo "[WARN] WANDB_API_KEY is not set. W&B online mode may fail."
fi

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
TEMPLATE_CFG="${TEMPLATE_CFG:-${PROJECT_DIR}/mf5/configs/mf5_train_lcofl_only.yaml}"

# LPR/LRPR 기본 데이터셋 규칙: trainval zip 사용
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-/data/pjh7639/datasets/raw-train-trainval.zip}"
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"      # 1: node local SSD staging
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"    # 1: keep local staged files
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

DATA_ROOT="${DATA_ROOT:-}"                 # used only when STAGE_TO_LOCAL=0

BASE_SR_CKPT="${BASE_SR_CKPT:-/data/pjh7639/weights/LCDNet/runs/lcdnet_2stage_exp01/stage2/best_model_cgnetV2_deformable_Epoch_8.pth}"
OCR_CKPT="${OCR_CKPT:-/data/pjh7639/weights/GP_LPR/runs/gplpr_2stage_66324_20260220_033315.pth}"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/pjh7639/weights/mf5}"
LOGS_ROOT="${LOGS_ROOT:-/data/pjh7639/logs}"
EXP_TAG="${EXP_TAG:-mf5_2stage_ab2b_lcofl_only}"

JOB_ID="${SLURM_JOB_ID:-manual}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${JOB_ID}_${TIMESTAMP}}"

RUN_ROOT="${RUN_ROOT:-${WEIGHTS_ROOT}/${EXP_TAG}/${RUN_ID}}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${LOGS_ROOT}/${EXP_TAG}/${RUN_ID}}"

CONFIG_DIR="${RUN_ROOT}/configs"
STAGE1_CFG="${CONFIG_DIR}/mf5_stage1_ab.yaml"
STAGE2_CFG="${CONFIG_DIR}/mf5_stage2_b.yaml"
STAGE1_SAVE_DIR="${RUN_ROOT}/stage1_ab"
STAGE2_SAVE_DIR="${RUN_ROOT}/stage2_b"
STAGE1_LOG_DIR="${RUN_LOG_ROOT}/stage1_ab"
STAGE2_LOG_DIR="${RUN_LOG_ROOT}/stage2_b"
PIPELINE_LOG="${RUN_LOG_ROOT}/pipeline.log"
RUN_SUMMARY_JSON="${RUN_LOG_ROOT}/run_summary.json"
EVAL_OUT_DIR="${RUN_LOG_ROOT}/val_ocr_eval_stage2"

SEED="${SEED:-1996}"
VAL_RATIO="${VAL_RATIO:-0.1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-40}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE1_LR="${STAGE1_LR:-}"                 # empty -> keep template value
STAGE2_LR="${STAGE2_LR:-}"                 # empty -> keep template value
WANDB_MODE="${WANDB_MODE:-}"               # empty -> keep template value
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-${EXP_TAG}_${RUN_ID}}"
RUN_OCR_EVAL="${RUN_OCR_EVAL:-0}"          # 1: run val OCR eval after stage2

mkdir -p "${RUN_ROOT}" "${RUN_LOG_ROOT}" "${CONFIG_DIR}" "${STAGE1_LOG_DIR}" "${STAGE2_LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

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

cleanup_local() {
  if [[ "${KEEP_LOCAL_DATA}" == "1" ]]; then
    return
  fi
  if [[ -n "${LOCAL_ROOT:-}" && -d "${LOCAL_ROOT}" ]]; then
    rm -rf "${LOCAL_ROOT}" || true
  fi
}
trap cleanup_local EXIT

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${TEMPLATE_CFG}" ]]; then
  echo "[ERROR] TEMPLATE_CFG not found: ${TEMPLATE_CFG}"
  exit 1
fi
for req in "${BASE_SR_CKPT}" "${OCR_CKPT}"; do
  if [[ ! -f "${req}" ]]; then
    echo "[ERROR] required checkpoint missing: ${req}"
    exit 1
  fi
done                                                                                                     

EFFECTIVE_DATA_ROOT=""
if [[ "${STAGE_TO_LOCAL}" == "1" ]]; then
  if [[ ! -f "${RAW_TRAIN_ZIP}" ]]; then
    echo "[ERROR] RAW_TRAIN_ZIP not found: ${RAW_TRAIN_ZIP}"
    exit 1
  fi

  if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
    echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
    LOCAL_BASE="${HOME}/scratch/pjh7639"
    mkdir -p "${LOCAL_BASE}"
  fi

  LOCAL_ROOT="${LOCAL_BASE}/mf5_2stage_ab2b_${RUN_ID}"
  LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
  LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
  mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}"

  echo "[INFO] staging train zip to local SSD: ${LOCAL_ROOT}"
  cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
  (
    cd "${LOCAL_RAW_DIR}"
    unzip -q -o "$(basename "${RAW_TRAIN_ZIP}")"
  )

  TRAIN_ROOT="$(resolve_train_root "${LOCAL_RAW_DIR}" || true)"
  if [[ -z "${TRAIN_ROOT}" ]]; then
    echo "[ERROR] failed to resolve train root from local staged zip."
    find "${LOCAL_RAW_DIR}" -maxdepth 4 -type d | sed -n '1,120p' || true
    exit 1
  fi
  ln -sfn "${TRAIN_ROOT}" "${LOCAL_DATA_DIR}/train"
  EFFECTIVE_DATA_ROOT="${LOCAL_DATA_DIR}"
else
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "[ERROR] set DATA_ROOT when STAGE_TO_LOCAL=0."
    exit 1
  fi
  if [[ ! -d "${DATA_ROOT}/train" ]]; then
    echo "[ERROR] DATA_ROOT/train not found: ${DATA_ROOT}/train"
    exit 1
  fi
  EFFECTIVE_DATA_ROOT="${DATA_ROOT}"
fi

echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] template cfg: ${TEMPLATE_CFG}"
echo "[INFO] data root: ${EFFECTIVE_DATA_ROOT}"
echo "[INFO] stage1 save: ${STAGE1_SAVE_DIR}"
echo "[INFO] stage2 save: ${STAGE2_SAVE_DIR}"
echo "[INFO] stage1 log: ${STAGE1_LOG_DIR}"
echo "[INFO] stage2 log: ${STAGE2_LOG_DIR}"

${PY_RUN[@]} - <<PY
import copy
import json
from pathlib import Path
import yaml

template_path = Path("${TEMPLATE_CFG}")
stage1_cfg_path = Path("${STAGE1_CFG}")
stage2_cfg_path = Path("${STAGE2_CFG}")
summary_path = Path("${RUN_SUMMARY_JSON}")

base = yaml.safe_load(template_path.read_text(encoding="utf-8"))

def apply_common(cfg):
    cfg["seed"] = int("${SEED}")
    cfg.setdefault("data", {})
    cfg["data"]["root"] = "${EFFECTIVE_DATA_ROOT}"
    cfg["data"]["val_ratio"] = float("${VAL_RATIO}")
    cfg["data"]["layout_filter"] = None

    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["input_mode"] = "stack15"

    cfg.setdefault("model", {})
    cfg.setdefault("model", {}).setdefault("args", {})
    cfg["model"]["args"]["in_channels"] = 15
    cfg["model"]["args"]["out_channels"] = 3

    cfg.setdefault("loss", {})
    cfg["loss"]["mode"] = "lcofl_only"

    cfg.setdefault("ocr", {})
    cfg["ocr"]["ocr_ckpt"] = "${OCR_CKPT}"
    cfg["ocr"]["ocr_train"] = False

    cfg.setdefault("adv", {})
    cfg["adv"]["enabled"] = False

    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = int("${BATCH_SIZE}")
    cfg["train"]["val_batch_size"] = int("${VAL_BATCH_SIZE}")
    cfg["train"]["num_workers"] = int("${NUM_WORKERS}")
    cfg["train"]["update_mode"] = "avg5_once"
    cfg["train"]["resume"] = None

    cfg.setdefault("wandb", {})
    if "${WANDB_MODE}":
        cfg["wandb"]["mode"] = "${WANDB_MODE}"

stage1 = copy.deepcopy(base)
apply_common(stage1)
stage1["model"]["checkpoint"] = "${BASE_SR_CKPT}"
stage1["data"]["scenario_filter"] = ["Scenario-A", "Scenario-B", "Senario-A", "Senario-B"]
stage1["train"]["epochs"] = int("${STAGE1_EPOCHS}")
stage1["train"]["save_dir"] = "${STAGE1_SAVE_DIR}"
stage1["train"]["log_dir"] = "${STAGE1_LOG_DIR}"
stage1["wandb"]["run_name"] = "${WANDB_RUN_PREFIX}_stage1_ab"
if "${STAGE1_LR}":
    stage1["train"]["lr"] = float("${STAGE1_LR}")

stage2 = copy.deepcopy(base)
apply_common(stage2)
stage2["model"]["checkpoint"] = "${STAGE1_SAVE_DIR}/best.pth"
stage2["data"]["scenario_filter"] = ["Scenario-B", "Senario-B"]
stage2["train"]["epochs"] = int("${STAGE2_EPOCHS}")
stage2["train"]["save_dir"] = "${STAGE2_SAVE_DIR}"
stage2["train"]["log_dir"] = "${STAGE2_LOG_DIR}"
stage2["wandb"]["run_name"] = "${WANDB_RUN_PREFIX}_stage2_b"
if "${STAGE2_LR}":
    stage2["train"]["lr"] = float("${STAGE2_LR}")

stage1_cfg_path.parent.mkdir(parents=True, exist_ok=True)
stage1_cfg_path.write_text(yaml.safe_dump(stage1, sort_keys=False), encoding="utf-8")
stage2_cfg_path.write_text(yaml.safe_dump(stage2, sort_keys=False), encoding="utf-8")

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary = {
    "project_dir": "${PROJECT_DIR}",
    "data_root": "${EFFECTIVE_DATA_ROOT}",
    "stage1": {
        "config": str(stage1_cfg_path),
        "save_dir": "${STAGE1_SAVE_DIR}",
        "log_dir": "${STAGE1_LOG_DIR}",
        "scenario_filter": stage1["data"]["scenario_filter"],
        "checkpoint_init": "${BASE_SR_CKPT}",
    },
    "stage2": {
        "config": str(stage2_cfg_path),
        "save_dir": "${STAGE2_SAVE_DIR}",
        "log_dir": "${STAGE2_LOG_DIR}",
        "scenario_filter": stage2["data"]["scenario_filter"],
        "checkpoint_init": "${STAGE1_SAVE_DIR}/best.pth",
    },
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[INFO] wrote {stage1_cfg_path}")
print(f"[INFO] wrote {stage2_cfg_path}")
print(f"[INFO] wrote {summary_path}")
PY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[INFO] validating stage1 config"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE1_CFG}" --validate-config-only

echo "[STAGE-1] training on Scenario-A + Scenario-B"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE1_CFG}"

if [[ ! -f "${STAGE1_SAVE_DIR}/best.pth" ]]; then
  echo "[ERROR] stage1 best checkpoint not found: ${STAGE1_SAVE_DIR}/best.pth"
  exit 1
fi

echo "[INFO] validating stage2 config"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE2_CFG}" --validate-config-only

echo "[STAGE-2] training on Scenario-B only"
${PY_RUN[@]} -m mf5.train_mf5 --config "${STAGE2_CFG}"

if [[ ! -f "${STAGE2_SAVE_DIR}/best.pth" ]]; then
  echo "[ERROR] stage2 best checkpoint not found: ${STAGE2_SAVE_DIR}/best.pth"
  exit 1
fi

if [[ "${RUN_OCR_EVAL}" == "1" ]]; then
  echo "[INFO] running optional val OCR evaluation for stage2 best checkpoint"
  ${PY_RUN[@]} -m mf5.eval_val_ocr_gplpr \
    --train-config "${STAGE2_CFG}" \
    --sr-checkpoint "${STAGE2_SAVE_DIR}/best.pth" \
    --ocr-checkpoint "${OCR_CKPT}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --batch-size "${VAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
fi

echo "[DONE] mf5 2-stage training completed"
echo "[DONE] stage1 best: ${STAGE1_SAVE_DIR}/best.pth"
echo "[DONE] stage2 best: ${STAGE2_SAVE_DIR}/best.pth"
echo "[DONE] pipeline log: ${PIPELINE_LOG}"
echo "[DONE] run summary: ${RUN_SUMMARY_JSON}"
