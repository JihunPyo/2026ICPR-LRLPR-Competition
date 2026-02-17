#!/bin/bash
#SBATCH --job-name=mf5_infer
#SBATCH --partition=batch_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

source ~/.bashrc
conda activate googlenet

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python mf5/infer_mf5.py \
  --config mf5/configs/mf5_infer.yaml \
  --checkpoint save/mf5_15ch/best.pth
