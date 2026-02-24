#!/bin/bash

#SBATCH -J mf5_2stage_ab2b_v2_20260223
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A.mf5_2stage_ab2b_v2_20260223.out

set -euo pipefail

# Example:
# sbatch --export=ALL,\
# PROJECT_DIR=/data/pjh7639/lpsr-lacd,\
# RUN_STAGE1=0,\
# STAGE1_BEST_CKPT=/data/pjh7639/weights/mf5/mf5_repro_65730_exp1/66569_20260222_181741/best.pth,\
# RAW_TRAIN_ZIP=/data/pjh7639/datasets/selected-train-trainval.zip,\
# OCR_CKPT=/data/pjh7639/weights/GP_LPR/gplpr_2stage_66324_20260220_033315_best_stage2.pth,\
# WANDB_API_KEY=xxxxx \
# scripts/slurm/train_mf5_2staged_ab2b_v2_20260223.slurm

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/train_mf5_2staged_ab2b_v2_20260223.sh
