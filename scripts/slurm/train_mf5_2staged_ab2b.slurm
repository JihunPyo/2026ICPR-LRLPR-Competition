#!/bin/bash

#SBATCH -J mf5_2stage_ab2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A.mf5_2stage_ab2b.out

set -euo pipefail

# Example:
# sbatch --export=ALL,PROJECT_DIR=/data/pjh7639/lpsr-lacd,RAW_TRAIN_ZIP=/data/pjh7639/datasets/raw-train-trainval.zip,BASE_SR_CKPT=/data/pjh7639/weights/LCDNet/runs/lcdnet_2stage_exp01/stage2/best_model_cgnetV2_deformable_Epoch_8.pth,OCR_CKPT=/data/pjh7639/weights/GP_LPR/runs/gplpr_2stage_66324_20260220_033315.pth scripts/slurm/train_mf5_2staged_ab2b.slurm

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/train_mf5_2staged_ab2b.sh
