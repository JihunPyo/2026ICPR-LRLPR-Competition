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

# Stage2 dataset is pinned to selected-train-trainval.zip.
REQUIRED_RAW_TRAIN_ZIP="/data/pjh7639/datasets/selected-train-trainval.zip"
RAW_TRAIN_ZIP="${RAW_TRAIN_ZIP:-${REQUIRED_RAW_TRAIN_ZIP}}"
if [[ "${RAW_TRAIN_ZIP}" != "${REQUIRED_RAW_TRAIN_ZIP}" ]]; then
  echo "[ERROR] RAW_TRAIN_ZIP must be ${REQUIRED_RAW_TRAIN_ZIP}"
  echo "[ERROR] current RAW_TRAIN_ZIP: ${RAW_TRAIN_ZIP}"
  exit 1
fi
STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-1}"      # 1: node local SSD staging
KEEP_LOCAL_DATA="${KEEP_LOCAL_DATA:-0}"    # 1: keep local staged files
LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

DATA_ROOT="${DATA_ROOT:-}"                 # used only when STAGE_TO_LOCAL=0

BASE_SR_CKPT="${BASE_SR_CKPT:-/data/pjh7639/weights/LCDNet/runs/lcdnet_2stage_exp01/stage2/best_model_cgnetV2_deformable_Epoch_8.pth}"
OCR_CKPT="${OCR_CKPT:-/data/pjh7639/weights/GP_LPR/gplpr_2stage_66324_20260220_033315_best_stage2.pth}"
RUN_STAGE1="${RUN_STAGE1:-0}"              # 1: run stage1(40ep) before stage2, 0: use STAGE1_BEST_CKPT
STAGE1_BEST_CKPT="${STAGE1_BEST_CKPT:-/data/pjh7639/weights/mf5/mf5_repro_65730_exp1/66569_20260222_181741/best.pth}"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/pjh7639/weights/mf5}"
LOGS_ROOT="${LOGS_ROOT:-/data/pjh7639/logs}"
EXP_TAG="${EXP_TAG:-mf5_2stage_selected_adv}"

JOB_ID="${SLURM_JOB_ID:-manual}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${JOB_ID}_${TIMESTAMP}}"

RUN_ROOT="${RUN_ROOT:-${WEIGHTS_ROOT}/${EXP_TAG}/${RUN_ID}}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${LOGS_ROOT}/${EXP_TAG}/${RUN_ID}}"

CONFIG_DIR="${RUN_ROOT}/configs"
STAGE1_CFG="${CONFIG_DIR}/mf5_stage1_readapt.yaml"
STAGE2_CFG="${CONFIG_DIR}/mf5_stage2_adv.yaml"
STAGE1_SAVE_DIR="${RUN_ROOT}/stage1_readapt"
STAGE2_SAVE_DIR="${RUN_ROOT}/stage2_adv"
STAGE1_LOG_DIR="${RUN_LOG_ROOT}/stage1_readapt"
STAGE2_LOG_DIR="${RUN_LOG_ROOT}/stage2_adv"
PIPELINE_LOG="${RUN_LOG_ROOT}/pipeline.log"
RUN_SUMMARY_JSON="${RUN_LOG_ROOT}/run_summary.json"
EVAL_OUT_DIR="${RUN_LOG_ROOT}/val_ocr_eval_stage2"

SEED="${SEED:-1996}"
VAL_RATIO="${VAL_RATIO:-0.1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-40}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
STAGE1_LR="${STAGE1_LR:-}"                 # empty -> keep template value
STAGE2_LR="${STAGE2_LR:-}"                 # empty -> keep template value
STAGE1_CE_WEIGHT="${STAGE1_CE_WEIGHT:-0.6}"
STAGE1_SSIM_WEIGHT="${STAGE1_SSIM_WEIGHT:-1.2}"
STAGE1_LAYOUT_WEIGHT="${STAGE1_LAYOUT_WEIGHT:-0.2}"
STAGE2_CE_WEIGHT="${STAGE2_CE_WEIGHT:-1.2}"
STAGE2_SSIM_WEIGHT="${STAGE2_SSIM_WEIGHT:-0.5}"
STAGE2_LAYOUT_WEIGHT="${STAGE2_LAYOUT_WEIGHT:-0.2}"
LOSS_MODE_STAGE1="${LOSS_MODE_STAGE1:-lcofl_only}"      # lcofl_only | losspack_mf5 | hybrid
LOSS_MODE_STAGE2="${LOSS_MODE_STAGE2:-lcofl_only}"      # lcofl_only | losspack_mf5 | hybrid
LOSSPACK_CE_IMPL_STAGE1="${LOSSPACK_CE_IMPL_STAGE1:-onehot}"  # onehot | logits
LOSSPACK_CE_IMPL_STAGE2="${LOSSPACK_CE_IMPL_STAGE2:-onehot}"  # onehot | logits
CM_ENABLED="${CM_ENABLED:-1}"
CM_THRESHOLD="${CM_THRESHOLD:-0.25}"
CM_PAIR_WEIGHT="${CM_PAIR_WEIGHT:-0.5}"
ADV_G_WEIGHT="${ADV_G_WEIGHT:-0.2}"
ADV_D_STEPS="${ADV_D_STEPS:-1}"
ADV_WARMUP_EPOCHS="${ADV_WARMUP_EPOCHS:-0}"
ADV_REAL_FRAME_MODE="${ADV_REAL_FRAME_MODE:-center}"
ADV_D_LR="${ADV_D_LR:-1.0e-4}"
ADV_D_BETAS="${ADV_D_BETAS:-[0.9, 0.999]}"
ADV_D_LR_STEP="${ADV_D_LR_STEP:-10}"
ADV_D_LR_GAMMA="${ADV_D_LR_GAMMA:-0.5}"
WANDB_MODE="${WANDB_MODE:-}"               # empty -> keep template value
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-${EXP_TAG}_${RUN_ID}}"
RUN_OCR_EVAL="${RUN_OCR_EVAL:-0}"          # 1: run val OCR eval after stage2

mkdir -p "${RUN_ROOT}" "${RUN_LOG_ROOT}" "${CONFIG_DIR}" "${STAGE1_LOG_DIR}" "${STAGE2_LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

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

if [[ "${RUN_STAGE1}" != "0" && "${RUN_STAGE1}" != "1" ]]; then
  echo "[ERROR] RUN_STAGE1 must be 0 or 1. current: ${RUN_STAGE1}"
  exit 1
fi

for m in "${LOSS_MODE_STAGE1}" "${LOSS_MODE_STAGE2}"; do
  case "${m}" in
    lcofl_only|losspack_mf5|hybrid) ;;
    *)
      echo "[ERROR] unsupported loss mode: ${m} (supported: lcofl_only|losspack_mf5|hybrid)"
      exit 1
      ;;
  esac
done

for c in "${LOSSPACK_CE_IMPL_STAGE1}" "${LOSSPACK_CE_IMPL_STAGE2}"; do
  case "${c}" in
    onehot|logits) ;;
    *)
      echo "[ERROR] unsupported losspack ce_impl: ${c} (supported: onehot|logits)"
      exit 1
      ;;
  esac
done

STAGE2_INIT_CKPT=""
if [[ "${RUN_STAGE1}" == "1" ]]; then
  STAGE2_INIT_CKPT="${STAGE1_SAVE_DIR}/best.pth"
else
  if [[ -z "${STAGE1_BEST_CKPT}" ]]; then
    echo "[ERROR] RUN_STAGE1=0 requires STAGE1_BEST_CKPT."
    exit 1
  fi
  if [[ ! -f "${STAGE1_BEST_CKPT}" ]]; then
    echo "[ERROR] STAGE1_BEST_CKPT not found: ${STAGE1_BEST_CKPT}"
    exit 1
  fi
  STAGE2_INIT_CKPT="${STAGE1_BEST_CKPT}"
fi

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

  LOCAL_ROOT="${LOCAL_BASE}/mf5_2stage_selected_adv_${RUN_ID}"
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
echo "[INFO] train zip: ${RAW_TRAIN_ZIP}"
echo "[INFO] data root: ${EFFECTIVE_DATA_ROOT}"
echo "[INFO] run stage1: ${RUN_STAGE1}"
echo "[INFO] stage1 loss mode: ${LOSS_MODE_STAGE1} (ce_impl=${LOSSPACK_CE_IMPL_STAGE1})"
echo "[INFO] stage2 loss mode: ${LOSS_MODE_STAGE2} (ce_impl=${LOSSPACK_CE_IMPL_STAGE2})"
echo "[INFO] stage2 init ckpt: ${STAGE2_INIT_CKPT}"
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
    cfg["data"]["scenario_filter"] = None
    cfg["data"]["layout_filter"] = None

    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["input_mode"] = "stack15"

    cfg.setdefault("model", {})
    cfg.setdefault("model", {}).setdefault("args", {})
    cfg["model"]["args"]["in_channels"] = 15
    cfg["model"]["args"]["out_channels"] = 3

    cfg.setdefault("loss", {})
    cfg["loss"]["lcofl_weight"] = 1.0
    cfg.setdefault("loss", {}).setdefault("lcofl", {})
    cfg.setdefault("loss", {}).setdefault("losspack_mf5", {})
    cfg["loss"]["confusion_matrix"] = {
        "enabled": bool(int("${CM_ENABLED}")),
        "threshold": float("${CM_THRESHOLD}"),
        "pair_weight": float("${CM_PAIR_WEIGHT}"),
    }

    cfg.setdefault("ocr", {})
    cfg["ocr"]["ocr_ckpt"] = "${OCR_CKPT}"
    cfg["ocr"]["ocr_train"] = False

    cfg["adv"] = {
        "enabled": False,
        "g_weight": 0.0,
        "d_steps": 0,
        "warmup_epochs": 0,
        "real_frame_mode": "center",
    }

    cfg.setdefault("train", {})
    cfg["train"]["batch_size"] = int("${BATCH_SIZE}")
    cfg["train"]["val_batch_size"] = int("${VAL_BATCH_SIZE}")
    cfg["train"]["num_workers"] = int("${NUM_WORKERS}")
    cfg["train"]["update_mode"] = "avg5_once"
    cfg["train"]["resume"] = None
    cfg.setdefault("train", {}).setdefault("val_ocr", {})
    cfg["train"]["val_ocr"]["enabled"] = True
    cfg["train"]["val_ocr"]["every_n_epochs"] = 1
    cfg["train"]["val_ocr"]["scope"] = "full_val"
    cfg["train"]["val_ocr"]["batch_size"] = int("${VAL_BATCH_SIZE}")
    cfg["train"]["val_ocr"]["num_workers"] = int("${NUM_WORKERS}")
    cfg["train"]["val_ocr"]["save_per_sample"] = False

    cfg.setdefault("wandb", {})
    if "${WANDB_MODE}":
        cfg["wandb"]["mode"] = "${WANDB_MODE}"

stage1 = copy.deepcopy(base)
apply_common(stage1)
stage1["model"]["checkpoint"] = "${BASE_SR_CKPT}"
stage1["train"]["epochs"] = int("${STAGE1_EPOCHS}")
stage1["train"]["save_dir"] = "${STAGE1_SAVE_DIR}"
stage1["train"]["log_dir"] = "${STAGE1_LOG_DIR}"
stage1["wandb"]["run_name"] = "${WANDB_RUN_PREFIX}_stage1_readapt"
stage1["loss"]["mode"] = "${LOSS_MODE_STAGE1}"
stage1["loss"].setdefault("losspack_mf5", {})
stage1["loss"]["losspack_mf5"]["ce_impl"] = "${LOSSPACK_CE_IMPL_STAGE1}"
stage1["loss"]["lcofl"].update(
    {
        "ce_weight": float("${STAGE1_CE_WEIGHT}"),
        "ssim_weight": float("${STAGE1_SSIM_WEIGHT}"),
        "layout_weight": float("${STAGE1_LAYOUT_WEIGHT}"),
    }
)
stage1["ocr"]["ocr_train"] = False
stage1["adv"]["enabled"] = False
if "${STAGE1_LR}":
    stage1["train"]["lr"] = float("${STAGE1_LR}")

stage2 = copy.deepcopy(base)
apply_common(stage2)
stage2["model"]["checkpoint"] = "${STAGE2_INIT_CKPT}"
stage2["train"]["epochs"] = int("${STAGE2_EPOCHS}")
stage2["train"]["save_dir"] = "${STAGE2_SAVE_DIR}"
stage2["train"]["log_dir"] = "${STAGE2_LOG_DIR}"
stage2["wandb"]["run_name"] = "${WANDB_RUN_PREFIX}_stage2_adv"
stage2["loss"]["mode"] = "${LOSS_MODE_STAGE2}"
stage2["loss"].setdefault("losspack_mf5", {})
stage2["loss"]["losspack_mf5"]["ce_impl"] = "${LOSSPACK_CE_IMPL_STAGE2}"
stage2["loss"]["lcofl"].update(
    {
        "ce_weight": float("${STAGE2_CE_WEIGHT}"),
        "ssim_weight": float("${STAGE2_SSIM_WEIGHT}"),
        "layout_weight": float("${STAGE2_LAYOUT_WEIGHT}"),
    }
)
stage2["ocr"]["ocr_train"] = True
stage2["adv"] = {
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
if "${STAGE2_LR}":
    stage2["train"]["lr"] = float("${STAGE2_LR}")

stage1_cfg_path.parent.mkdir(parents=True, exist_ok=True)
stage1_cfg_path.write_text(yaml.safe_dump(stage1, sort_keys=False), encoding="utf-8")
stage2_cfg_path.write_text(yaml.safe_dump(stage2, sort_keys=False), encoding="utf-8")

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary = {
    "project_dir": "${PROJECT_DIR}",
    "data_root": "${EFFECTIVE_DATA_ROOT}",
    "run_stage1": bool(int("${RUN_STAGE1}")),
    "stage2_init_ckpt": "${STAGE2_INIT_CKPT}",
    "loss_ratio": {
        "stage1": {
            "ce_weight": float("${STAGE1_CE_WEIGHT}"),
            "ssim_weight": float("${STAGE1_SSIM_WEIGHT}"),
            "layout_weight": float("${STAGE1_LAYOUT_WEIGHT}"),
        },
        "stage2": {
            "ce_weight": float("${STAGE2_CE_WEIGHT}"),
            "ssim_weight": float("${STAGE2_SSIM_WEIGHT}"),
            "layout_weight": float("${STAGE2_LAYOUT_WEIGHT}"),
        },
    },
    "stage1": {
        "config": str(stage1_cfg_path),
        "save_dir": "${STAGE1_SAVE_DIR}",
        "log_dir": "${STAGE1_LOG_DIR}",
        "scenario_filter": stage1["data"].get("scenario_filter"),
        "checkpoint_init": "${BASE_SR_CKPT}",
        "loss_mode": stage1["loss"].get("mode"),
        "losspack_ce_impl": stage1["loss"].get("losspack_mf5", {}).get("ce_impl"),
    },
    "stage2": {
        "config": str(stage2_cfg_path),
        "save_dir": "${STAGE2_SAVE_DIR}",
        "log_dir": "${STAGE2_LOG_DIR}",
        "scenario_filter": stage2["data"].get("scenario_filter"),
        "checkpoint_init": "${STAGE2_INIT_CKPT}",
        "loss_mode": stage2["loss"].get("mode"),
        "losspack_ce_impl": stage2["loss"].get("losspack_mf5", {}).get("ce_impl"),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"[INFO] wrote {stage1_cfg_path}")
print(f"[INFO] wrote {stage2_cfg_path}")
print(f"[INFO] wrote {summary_path}")
PY

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[INFO] validating stage1 config"
  ${PY_RUN[@]} -m mf5.train_mf5_20260223 --config "${STAGE1_CFG}" --validate-config-only

  echo "[STAGE-1] readaptation training (LCOFL, CE/SSIM weighted)"
  ${PY_RUN[@]} -m mf5.train_mf5_20260223 --config "${STAGE1_CFG}"

  if [[ ! -f "${STAGE1_SAVE_DIR}/best.pth" ]]; then
    echo "[ERROR] stage1 best checkpoint not found: ${STAGE1_SAVE_DIR}/best.pth"
    exit 1
  fi
else
  echo "[INFO] skipping stage1 (RUN_STAGE1=0). use checkpoint: ${STAGE2_INIT_CKPT}"
fi

echo "[INFO] validating stage2 config"
${PY_RUN[@]} -m mf5.train_mf5_20260223 --config "${STAGE2_CFG}" --validate-config-only

echo "[STAGE-2] adversarial training (OCR unfreeze + LCOFL fixed)"
${PY_RUN[@]} -m mf5.train_mf5_20260223 --config "${STAGE2_CFG}"

if [[ ! -f "${STAGE2_SAVE_DIR}/best.pth" ]]; then
  echo "[ERROR] stage2 best checkpoint not found: ${STAGE2_SAVE_DIR}/best.pth"
  exit 1
fi

if [[ "${RUN_OCR_EVAL}" == "1" ]]; then
  echo "[INFO] running optional val OCR evaluation for stage2 best checkpoint"
  ${PY_RUN[@]} -m mf5.eval_val_ocr_gplpr_20260223 \
    --train-config "${STAGE2_CFG}" \
    --sr-checkpoint "${STAGE2_SAVE_DIR}/best.pth" \
    --ocr-checkpoint "${OCR_CKPT}" \
    --output-dir "${EVAL_OUT_DIR}" \
    --batch-size "${VAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}"
fi

echo "[DONE] mf5 2-stage training completed"
if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "[DONE] stage1 best: ${STAGE1_SAVE_DIR}/best.pth"
else
  echo "[DONE] stage1 best (input): ${STAGE2_INIT_CKPT}"
fi
echo "[DONE] stage2 best: ${STAGE2_SAVE_DIR}/best.pth"
echo "[DONE] pipeline log: ${PIPELINE_LOG}"
echo "[DONE] run summary: ${RUN_SUMMARY_JSON}"
