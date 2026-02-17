#!/bin/bash
#SBATCH --job-name=mf5_s2_ocr
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/pjh7639/lpsr-lacd
#SBATCH --output=/data/pjh7639/lpsr-lacd/%x_%j.out

set -euo pipefail

# Usage:
# sbatch --export=ALL,RUN_ROOT=/data/pjh7639/datasets/mf5_runs/adv_20260216_043933,OCR_CKPT=/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth scripts/slurm/infer_mf5_stage2_ocr.slurm
#
# Optional overrides:
# PROJECT_DIR=/data/pjh7639/lpsr-lacd
# STAGE2_CKPT=/data/.../stage2_adv/best.pth
# INFER_CFG=/data/.../configs/mf5_infer_test.yaml
# SR_DIR=/data/.../test_sr
# OCR_TXT=/data/.../test_sr_ocr_predictions.txt
# OCR_BATCH_SIZE=64
# NUM_WORKERS=8

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/infer_mf5_stage2_ocr.sh
