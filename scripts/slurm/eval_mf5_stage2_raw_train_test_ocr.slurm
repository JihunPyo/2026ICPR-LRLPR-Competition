#!/bin/bash

#SBATCH -J mf5_s2_rawtest_ocr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 0-12:00:00
#SBATCH -o slurm-%A.mf5_s2_rawtest_ocr.out

set -euo pipefail

# Example:
# sbatch --export=ALL,\
# PROJECT_DIR=/data/pjh7639/lpsr-lacd,\
# RUN_ROOT=/data/pjh7639/weights/mf5/mf5_2stage_selected_adv/66569_20260223_231500,\
# RAW_TEST_ZIP=/data/pjh7639/datasets/raw-train-test.zip,\
# STAGE_TO_LOCAL=1 \
# scripts/slurm/eval_mf5_stage2_raw_train_test_ocr.slurm
#
# Optional:
# STAGE2_CKPT=/data/pjh7639/weights/mf5/.../stage2_adv/best.pth
# TRAIN_CFG=/data/pjh7639/weights/mf5/.../configs/mf5_stage2_adv.yaml
# OCR_CKPT=/data/pjh7639/weights/GP_LPR/gplpr_2stage_66324_20260220_033315_best_stage2.pth
# OUTPUT_ROOT=/data/pjh7639/weights/mf5/evals
# NAS_SR_ZIP=/data/pjh7639/datasets/mf5-sr/sr-test-mf5_v2.zip
# RUN_TAG=manual_eval_001
# BATCH_SIZE=32 NUM_WORKERS=8

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/eval_mf5_stage2_raw_train_test_ocr.sh
