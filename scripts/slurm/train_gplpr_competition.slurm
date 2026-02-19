#!/bin/bash
#SBATCH --job-name=gplpr_comp
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
CONFIG="${CONFIG:-configs/gplpr_train_competition.yaml}"
SPLIT_NAS="${SPLIT_NAS:-/data/pjh7639/lpsr-lacd/split_all_rodosol_preprocessed.txt}"
PREPROC_ROOT_NAS="${PREPROC_ROOT_NAS:-/data/pjh7639/datasets/rodosol_preprocessed_16x48_32x96}"
PREPROC_TAR_NAS="${PREPROC_TAR_NAS:-/data/pjh7639/datasets/rodosol_preprocessed_16x48_32x96.tar}"
LOCAL_BASE="${LOCAL_BASE:-/local_datasets/pjh7639}"
LOCAL_PREPROC_ROOT="${LOCAL_PREPROC_ROOT:-${LOCAL_BASE}/rodosol_preprocessed_16x48_32x96}"
LOCAL_SPLIT="${LOCAL_SPLIT:-${LOCAL_BASE}/split_all_rodosol_preprocessed.local.txt}"
LOCAL_CONFIG="${LOCAL_CONFIG:-${LOCAL_BASE}/gplpr_train_competition.local.yaml}"
export PROJECT_DIR CONFIG LOCAL_SPLIT LOCAL_CONFIG

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] config: ${CONFIG}"
echo "[INFO] split (nas): ${SPLIT_NAS}"
echo "[INFO] preprocessed root (nas): ${PREPROC_ROOT_NAS}"
echo "[INFO] preprocessed tar (nas): ${PREPROC_TAR_NAS}"
echo "[INFO] preprocessed root (local): ${LOCAL_PREPROC_ROOT}"

cd "${PROJECT_DIR}"

mkdir -p "${LOCAL_BASE}"
rm -rf "${LOCAL_PREPROC_ROOT}"
mkdir -p "${LOCAL_PREPROC_ROOT}"

echo "[INFO] staging preprocessed dataset via tar..."
if [[ -f "${PREPROC_TAR_NAS}" ]]; then
  cp "${PREPROC_TAR_NAS}" "${LOCAL_BASE}/preprocessed.tar"
  tar -xf "${LOCAL_BASE}/preprocessed.tar" -C "${LOCAL_BASE}"
else
  echo "[WARN] tar not found, fallback to directory copy"
  cp -a "${PREPROC_ROOT_NAS}/." "${LOCAL_PREPROC_ROOT}/"
fi

echo "[INFO] rewriting split paths for local dataset..."
sed "s#${PREPROC_ROOT_NAS}#${LOCAL_PREPROC_ROOT}#g" "${SPLIT_NAS}" > "${LOCAL_SPLIT}"

echo "[INFO] creating local training config..."
mkdir -p "$(dirname "${LOCAL_CONFIG}")"
cp "${PROJECT_DIR}/${CONFIG}" "${LOCAL_CONFIG}"
# Replace both relative and absolute split paths if present.
sed -i "s#\./split_all_rodosol_preprocessed.txt#${LOCAL_SPLIT}#g" "${LOCAL_CONFIG}"
sed -i "s#/data/pjh7639/lpsr-lacd/split_all_rodosol_preprocessed.txt#${LOCAL_SPLIT}#g" "${LOCAL_CONFIG}"
echo "[INFO] local config saved: ${LOCAL_CONFIG}"

echo "[INFO] start training with local staged dataset"
conda run -n googlenet python -m train_gplpr --config "${LOCAL_CONFIG}"
``