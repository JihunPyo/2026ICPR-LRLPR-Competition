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
- Loss modes:
  - `hybrid`: pixel (`L1 + SSIM`) + optional OCR-guided terms.
  - `lcofl_only`: 3-term LCOFL only (`CE + SSIM + layout`), GPLPR frozen.
  - `losspack_mf5`: lossPack-style 3-term OCR-perceptual loss (`CE + SSIM + layout`) with MF5-compatible data flow.

## Implementation summary

- MF5 does not introduce a separate SR model class.
  - Training/inference still instantiate `cgnetV2_deformable` via `models.make(...)`.
- MF5-specific adaptation is handled by configuration + pipeline:
  - `input_mode=stack15` -> model `in_channels=15`
  - `input_mode=center3` -> model `in_channels=3`
  - residual skip for multi-frame input uses center RGB triplet inside `cgnet_deformable2d_arch.py`.
- Training unit is track-level (5-frame LR/HR pairs), not single-frame.
  - Wrapper outputs `lr: [B,15,H,W]` or `[B,3,H,W]`
  - Wrapper outputs `hr: [B,5,3,2H,2W]` for train/val.
- Losses:
  - base pixel loss: `L1 + SSIM`
  - optional OCR-guided `LCOFL`
  - optional adversarial-like OCR update when `adv.enabled=true`.
- `lcofl_only` mode uses local `kornia.SSIMLoss` wrapper (no `lossPack.py` import).
- `losspack_mf5` mode keeps lossPack philosophy while supporting MF5 training loop:
  - `loss.losspack_mf5.ce_impl=onehot` (default, original-style behavior)
  - `loss.losspack_mf5.ce_impl=logits` (differentiable CE path)
- In `lcofl_only`, OCR CE is class-weighted (`w_k`) and updated from validation confusion pairs.

## Dependencies

```bash
pip install -r mf5/requirements_mf5.txt
```

## Config validation

Use a fast config-only check before training:

```bash
python -m mf5.train_mf5 --config mf5/configs/mf5_train.yaml --validate-config-only
python -m mf5.train_mf5 --config mf5/configs/mf5_train_baseline_center3.yaml --validate-config-only
python -m mf5.train_mf5 --config mf5/configs/mf5_train_lcofl_only.yaml --validate-config-only
```

The validator checks:

- `preprocess.input_mode` and `model.args.in_channels` consistency.
- `train.update_mode` validity (`avg5_once` or `per_hr_step`).
- OCR checkpoint requirement when `lcofl_weight > 0` or `adv.enabled=true`.
- `lcofl_only` constraints:
  - `model.checkpoint` required
  - `ocr.ocr_ckpt` required
  - `ocr.ocr_train=false` required
  - `adv.enabled=false` required
- `losspack_mf5` constraints:
  - `model.checkpoint` required
  - `ocr.ocr_ckpt` required
  - `loss.losspack_mf5.ce_impl` in `{onehot, logits}`
  - TensorFlow/Keras options are not supported in MF5 training path

## LCOFL-Only Training

Use:

```bash
python -m mf5.train_mf5 --config mf5/configs/mf5_train_lcofl_only.yaml
```

Important:

- Set `model.checkpoint` to your competition-trained LCDNet SR checkpoint before running.
- GPLPR is used only as a frozen OCR scorer (`ocr_train=false`).
- `loss.confusion_matrix.enabled=true` is recommended (default in `mf5_train_lcofl_only.yaml`) to update `w_k`.
- 3-term default ratio:
  - CE: `1.0`
  - SSIM: `0.75`
  - layout: `0.2`
- Early stopping monitors `val_loss` with:
  - `patience=8`, `min_delta=1e-4`, `min_epochs=10`.

## Run

```bash
python -m mf5.train_mf5 --config mf5/configs/mf5_train.yaml
python -m mf5.infer_mf5 --config mf5/configs/mf5_infer.yaml --checkpoint save/mf5_15ch/best.pth
```

Baseline comparison run:

```bash
python -m mf5.train_mf5 --config mf5/configs/mf5_train_baseline_center3.yaml
```

For Slurm, use:

- `scripts/slurm/train_mf5.slurm`
- `scripts/slurm/infer_mf5.slurm`

## GPLPR HR5 OCR Evaluation (raw-train-trainval)

Evaluate all HR-001..005 images per track with GPLPR OCR and build reduction manifests.

Local command:

```bash
python -m mf5.eval_trainval_hr5_ocr_gplpr \
  --input-zip /data/pjh7639/datasets/raw-train-trainval.zip \
  --ocr-checkpoint /data/pjh7639/weights/GP_LPR/Rodosol.pth \
  --output-dir /data/pjh7639/weights/GP_LPR/evals/hr5_eval_run01
```

Or with extracted train root:

```bash
python -m mf5.eval_trainval_hr5_ocr_gplpr \
  --train-root /data/pjh7639/datasets/raw_train_unzip \
  --ocr-checkpoint /data/pjh7639/weights/GP_LPR/Rodosol.pth \
  --output-dir /data/pjh7639/weights/GP_LPR/evals/hr5_eval_run01
```

Outputs:

- `per_image_ocr.csv`: OCR result + confidence for every HR image
- `per_track_summary.csv`: track-level hit flag and best-correct frame info
- `selected_tracks_manifest.csv`: tracks to keep (`LR all + selected HR 1 frame`)
- `dropped_tracks.csv`: tracks with no correct HR among 5
- `scenario_summary.csv`: A/B/overall/macro-average track hit rates
- `run_summary.json`: includes
  - `scenario_a_track_hit_rate`
  - `scenario_b_track_hit_rate`
  - `scenario_macro_avg_track_hit_rate`
  - `overall_track_hit_rate`

Slurm command:

```bash
sbatch --export=ALL,PROJECT_DIR=/data/pjh7639/lpsr-lacd,OCR_CKPT=/data/pjh7639/weights/GP_LPR/<your_ckpt>.pth scripts/slurm/eval_gplpr_hr5_trainval.slurm
```

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
