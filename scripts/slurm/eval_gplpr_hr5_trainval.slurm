#!/bin/bash
#SBATCH --job-name=gplpr_hr5_eval
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/data/pjh7639/lpsr-lacd
#SBATCH --output=/data/pjh7639/lpsr-lacd/%x_%j.out

set -euo pipefail

# Example:
# sbatch --export=ALL,PROJECT_DIR=/data/pjh7639/lpsr-lacd,OCR_CKPT=/data/pjh7639/weights/GP_LPR/<your_ckpt>.pth scripts/slurm/eval_gplpr_hr5_trainval.slurm

PROJECT_DIR="${PROJECT_DIR:-/data/pjh7639/lpsr-lacd}"
cd "${PROJECT_DIR}"

bash scripts/slurm/eval_gplpr_hr5_trainval.sh
