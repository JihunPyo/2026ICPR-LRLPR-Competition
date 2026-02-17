#!/bin/bash
#SBATCH --job-name=mf5_adv_pipe
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Example:
# sbatch --export=ALL,PROJECT_DIR=/data/pjh7639/lpsr-lacd,DATA_ROOT=/data/pjh7639/datasets/lpr_data,BASE_MF5_CKPT=/data/pjh7639/datasets/mf5_runs/mf5_12345/best.pth,OCR_CKPT=/data/pjh7639/lpsr-lacd/save/gplpr_competition/best_gplpr.pth scripts/slurm/train_mf5_adv_pipeline.slurm

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/train_mf5_adv_pipeline.sh
