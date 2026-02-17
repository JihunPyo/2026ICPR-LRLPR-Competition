# MF5 (5-frame LR -> 15-channel) pipeline

This module adds a separate training/inference path for multi-frame LP super-resolution:

- Input:
  - `stack15`: 5 LR frames concatenated into 15 channels.
  - `center3`: center LR frame only (baseline comparison).
- Target (train/val): 5 HR frames per track.
- Output: 1 SR image per track.

## Key behavior

- Preprocessing follows the original project wrapper logic:
  - aspect-ratio padding
  - bicubic resize
  - LR-only augmentation
  - CLAHE on HR
- Training update modes:
  - `avg5_once`: one forward, average loss against HR1..HR5, one optimizer step.
  - `per_hr_step`: for one LR15 input, update 5 times using HR1..HR5 sequentially.

## Dependencies

```bash
pip install -r mf5/requirements_mf5.txt
```

## Run

```bash
python mf5/train_mf5.py --config mf5/configs/mf5_train.yaml
python mf5/infer_mf5.py --config mf5/configs/mf5_infer.yaml --checkpoint save/mf5_15ch/best.pth
```

Baseline comparison run:

```bash
python mf5/train_mf5.py --config mf5/configs/mf5_train_baseline_center3.yaml
```

For Slurm, use:

- `scripts/slurm/train_mf5.slurm`
- `scripts/slurm/infer_mf5.slurm`

## Weights & Biases (optional)

Set in `mf5/configs/mf5_train.yaml`:

```yaml
wandb:
  enabled: true
  project: lpr-mf5
  entity: <your-team-or-user>
  run_name: mf5-15ch-ft
  mode: online
```

If you want to run without internet access on cluster nodes, use:

```yaml
wandb:
  enabled: true
  mode: offline
```
