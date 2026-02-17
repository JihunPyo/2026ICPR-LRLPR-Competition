#!/bin/bash
#SBATCH --job-name=mf5_infer_one
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Usage:
#   sbatch scripts/slurm/infer_mf5_one.slurm
# Optional overrides:
#   sbatch --export=ALL,PROJECT_DIR=/data/pjh7639/lpsr-lacd,CHECKPOINT=/data/pjh7639/datasets/mf5_runs/mf5_65680/best.pth,CONFIG=mf5/configs/mf5_infer.yaml scripts/slurm/infer_mf5_one.slurm

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
CHECKPOINT="${CHECKPOINT:-/data/pjh7639/datasets/mf5_runs/mf5_65680/best.pth}"
CONFIG="${CONFIG:-mf5/configs/mf5_infer.yaml}"

echo "[INFO] host: $(hostname)"
echo "[INFO] project dir: ${PROJECT_DIR}"
echo "[INFO] checkpoint: ${CHECKPOINT}"
echo "[INFO] config: ${CONFIG}"

cd "${PROJECT_DIR}"

conda run -n googlenet python -m mf5.infer_mf5 \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}"
