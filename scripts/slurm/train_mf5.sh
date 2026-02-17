#!/bin/bash

#SBATCH -J lpr_mf5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad
#SBATCH -t 0-12:00:00
#SBATCH -o slurm-%A.bootstrap.out

set -euo pipefail

# 1. 가상환경 활성화
source ~/.bashrc || true

# sbatch 비대화형 셸에서도 conda activate가 동작하도록 보강
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda not found in this batch environment."
  exit 1
fi

conda activate googlenet || {
  echo "[ERROR] failed to activate conda env: googlenet"
  conda info --envs || true
  exit 1
}

# Optional: W&B login
# 제출 예시:
# sbatch --export=ALL,WANDB_API_KEY=xxxxx scripts/slurm/train_mf5.sh
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] WANDB_API_KEY detected. Logging in to W&B."
  wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "[WARN] WANDB_API_KEY is not set. Use offline mode or provide key."
fi

# 2. 경로 설정
# 기본값은 sbatch를 실행한 디렉토리(레포 루트)로 둔다.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NAS_DATA_ROOT="/data/pjh7639/datasets"
RAW_TRAIN_ZIP="${NAS_DATA_ROOT}/raw-train.zip"
RAW_TEST_ZIP="${NAS_DATA_ROOT}/raw-test-public.zip"
PRETRAINED_CKPT="/data/pjh7639/datasets/best_model_cgnetV2_deformable_Epoch_82.pth"
OCR_CKPT="/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth"
NAS_OUT_ROOT="/data/pjh7639/datasets/mf5_runs"
RUN_OCR_EVAL="${RUN_OCR_EVAL:-0}"  # 0: skip (default), 1: run val OCR eval after training

LOCAL_BASE_DEFAULT="/local_datasets/pjh7639"
LOCAL_BASE="${LOCAL_BASE:-${LOCAL_BASE_DEFAULT}}"

# 복사 대상 상위 디렉토리를 먼저 준비한다.
if ! mkdir -p "${LOCAL_BASE}" 2>/dev/null; then
  echo "[WARN] cannot create ${LOCAL_BASE}. fallback to \$HOME/scratch."
  LOCAL_BASE="${HOME}/scratch/pjh7639"
  mkdir -p "${LOCAL_BASE}"
fi

LOCAL_ROOT="${LOCAL_BASE}/lpr_mf5_${SLURM_JOB_ID}"
LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
RUNTIME_CFG="${LOCAL_ROOT}/mf5_train.runtime.yaml"
RUN_OUT_DIR="${NAS_OUT_ROOT}/mf5_${SLURM_JOB_ID}"
RUN_CFG_COPY="${RUN_OUT_DIR}/mf5_train.runtime.yaml"
EVAL_OUT_DIR="${RUN_OUT_DIR}/val_ocr_eval"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}" "${RUN_OUT_DIR}"

# 메인 로그는 프로젝트 logs 디렉토리로 남긴다.
LOG_FILE="${PROJECT_DIR}/logs/slurm-${SLURM_JOB_ID}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[INFO] log file: ${LOG_FILE}"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] run output dir: ${RUN_OUT_DIR}"
echo "[INFO] local base dir: ${LOCAL_BASE}"
echo "[INFO] RUN_OCR_EVAL: ${RUN_OCR_EVAL}"
echo "[INFO] host: $(hostname)"
echo "[INFO] user: $(whoami)"

# 2-1. 사전 점검: 노드 디렉토리/권한/입력 파일
if [[ -d /local_datasets ]]; then
  ls -ld /local_datasets || true
else
  echo "[WARN] /local_datasets does not exist on this node."
fi

if [[ ! -w "${LOCAL_BASE}" ]]; then
  echo "[ERROR] local base is not writable: ${LOCAL_BASE}"
  exit 1
fi

for required_file in "${RAW_TRAIN_ZIP}" "${RAW_TEST_ZIP}" "${PRETRAINED_CKPT}" "${OCR_CKPT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[ERROR] required file not found: ${required_file}"
    exit 1
  fi
done

# 3. 로컬 SSD로 데이터 복사 후 압축 해제
cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
cp "${RAW_TEST_ZIP}" "${LOCAL_RAW_DIR}/"
cd "${LOCAL_RAW_DIR}"
unzip -q -o raw-train.zip
unzip -q -o raw-test-public.zip

# 4. 학습 코드가 기대하는 구조(data/train, data/test-public)로 정렬
TRAIN_PARENT="$(dirname "$(find "${LOCAL_RAW_DIR}" -type d -name 'Scenario-A' | head -n1)")"
TEST_PARENT="$(dirname "$(dirname "$(find "${LOCAL_RAW_DIR}" -type f -name 'lr-001.jpg' | head -n1)")")"
if [[ -z "${TRAIN_PARENT}" || -z "${TEST_PARENT}" ]]; then
  echo "[ERROR] failed to detect train/test roots after unzip."
  exit 1
fi
ln -sfn "${TRAIN_PARENT}" "${LOCAL_DATA_DIR}/train"
ln -sfn "${TEST_PARENT}" "${LOCAL_DATA_DIR}/test-public"

# 5. 런타임 config 생성 (로컬 데이터 + 파인튜닝 체크포인트 반영)
python - <<PY
import yaml
from pathlib import Path

src = Path("${PROJECT_DIR}") / "mf5/configs/mf5_train_lcofl.yaml"
dst = Path("${RUNTIME_CFG}")
cfg = yaml.safe_load(src.read_text())
cfg["data"]["root"] = "${LOCAL_DATA_DIR}"
cfg["model"]["checkpoint"] = "${PRETRAINED_CKPT}"
cfg["ocr"]["ocr_ckpt"] = "${OCR_CKPT}"
cfg["train"]["save_dir"] = "${RUN_OUT_DIR}"
cfg["train"]["update_mode"] = "avg5_once"
cfg["train"]["num_workers"] = 8
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"saved runtime config: {dst}")
PY
cp "${RUNTIME_CFG}" "${RUN_CFG_COPY}"
echo "[INFO] runtime config copied: ${RUN_CFG_COPY}"

# 6. 프로젝트 폴더로 이동 후 학습 실행
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
python -u mf5/train_mf5.py --config "${RUNTIME_CFG}"

# 6-1. 학습 완료 후 검증셋 OCR 정확도 평가 (optional)
if [[ "${RUN_OCR_EVAL}" == "1" ]]; then
  BEST_SR_CKPT="${RUN_OUT_DIR}/best.pth"
  if [[ -f "${BEST_SR_CKPT}" ]]; then
    echo "[INFO] running val OCR evaluation..."
    python -u mf5/eval_val_ocr_gplpr.py \
      --train-config "${RUNTIME_CFG}" \
      --sr-checkpoint "${BEST_SR_CKPT}" \
      --ocr-checkpoint "${OCR_CKPT}" \
      --output-dir "${EVAL_OUT_DIR}" \
      --batch-size 32 \
      --num-workers 8
    echo "[INFO] val OCR summary: ${EVAL_OUT_DIR}/val_ocr_summary.json"
  else
    echo "[WARN] best checkpoint not found, skip OCR evaluation: ${BEST_SR_CKPT}"
  fi
else
  echo "[INFO] RUN_OCR_EVAL=0, skip val OCR evaluation."
fi

# 7. 작업 종료 후 로컬 데이터 삭제
rm -rf "${LOCAL_ROOT}"
