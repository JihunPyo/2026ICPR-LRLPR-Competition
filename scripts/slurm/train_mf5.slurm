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
source ~/.bashrc
conda activate googlenet

# Optional: W&B login
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[INFO] WANDB_API_KEY detected. Logging in to W&B."
  wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "[WARN] WANDB_API_KEY is not set. Use offline mode or provide key."
fi

# 2. 경로 설정
PROJECT_DIR="/data/pjh7639/lpsr-lacd"  # 필요시 수정
NAS_DATA_ROOT="/data/pjh7639/datasets"
RAW_TRAIN_ZIP="${NAS_DATA_ROOT}/raw-train.zip"
RAW_TEST_ZIP="${NAS_DATA_ROOT}/raw-test-public.zip"
PRETRAINED_CKPT="/data/pjh7639/datasets/best_model_cgnetV2_deformable_Epoch_82.pth"
NAS_OUT_ROOT="/data/pjh7639/datasets/mf5_runs"

LOCAL_ROOT="/local_datasets/pjh7639/lpr_mf5_${SLURM_JOB_ID}"
LOCAL_RAW_DIR="${LOCAL_ROOT}/raw"
LOCAL_DATA_DIR="${LOCAL_ROOT}/data"
RUNTIME_CFG="${LOCAL_ROOT}/mf5_train.runtime.yaml"
RUN_OUT_DIR="${NAS_OUT_ROOT}/mf5_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${LOCAL_RAW_DIR}" "${LOCAL_DATA_DIR}" "${RUN_OUT_DIR}"

LOG_FILE="${PROJECT_DIR}/logs/slurm-${SLURM_JOB_ID}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[INFO] log file: ${LOG_FILE}"
echo "[INFO] run output dir: ${RUN_OUT_DIR}"

# 3. 로컬 SSD로 데이터 복사 후 압축 해제
cp "${RAW_TRAIN_ZIP}" "${LOCAL_RAW_DIR}/"
cp "${RAW_TEST_ZIP}" "${LOCAL_RAW_DIR}/"
cd "${LOCAL_RAW_DIR}"
unzip -q -o raw-train.zip
unzip -q -o raw-test-public.zip

# 4. 학습 코드가 기대하는 구조(data/train, data/test-public)로 정렬
TRAIN_PARENT="$(dirname "$(find "${LOCAL_RAW_DIR}" -type d -name 'Scenario-A' | head -n1)")"
TEST_PARENT="$(dirname "$(dirname "$(find "${LOCAL_RAW_DIR}" -type f -name 'lr-001.jpg' | head -n1)")")"
ln -sfn "${TRAIN_PARENT}" "${LOCAL_DATA_DIR}/train"
ln -sfn "${TEST_PARENT}" "${LOCAL_DATA_DIR}/test-public"

# 5. 런타임 config 생성 (로컬 데이터 + 파인튜닝 체크포인트 반영)
python - <<PY
import yaml
from pathlib import Path

src = Path("${PROJECT_DIR}") / "mf5/configs/mf5_train.yaml"
dst = Path("${RUNTIME_CFG}")
cfg = yaml.safe_load(src.read_text())
cfg["data"]["root"] = "${LOCAL_DATA_DIR}"
cfg["model"]["checkpoint"] = "${PRETRAINED_CKPT}"
cfg["train"]["save_dir"] = "${RUN_OUT_DIR}"
cfg["train"]["update_mode"] = "avg5_once"
cfg["train"]["num_workers"] = 8
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"saved runtime config: {dst}")
PY

# 6. 프로젝트 폴더로 이동 후 학습 실행
cd "${PROJECT_DIR}"
python -u mf5/train_mf5.py --config "${RUNTIME_CFG}"

# 7. 작업 종료 후 로컬 데이터 삭제
rm -rf "${LOCAL_ROOT}"
