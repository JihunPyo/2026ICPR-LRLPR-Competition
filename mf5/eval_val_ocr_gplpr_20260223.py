import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from mf5.data_20260223 import TrackSequenceDataset, TrackSequenceWrapper
from train_funcs.train_utils import strLabelConverter


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def clean_plate_text(text: str) -> str:
    return text.replace("#", "").replace("-", "").strip()


def normalize_state_dict_keys(sd: dict):
    if not sd:
        return sd
    has_module_prefix = all(k.startswith("module.") for k in sd.keys())
    if not has_module_prefix:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def load_sr_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in ckpt or "sd" not in ckpt["model"]:
        raise ValueError(f"Unsupported SR checkpoint format: {checkpoint_path}")

    spec = ckpt["model"]
    model = models.make({"name": spec["name"], "args": spec["args"]})
    model.load_state_dict(normalize_state_dict_keys(spec["sd"]), strict=True)
    model = model.to(device)
    model.eval()
    return model, spec["args"]


def load_gplpr_model(
    checkpoint_path: Path,
    device: torch.device,
    alphabet: str,
    nc: int,
    k: int,
    is_seq_model: bool,
    head: int,
    inner: int,
    is_l2_norm: bool,
):
    model_args = {
        "alphabet": alphabet,
        "nc": nc,
        "K": k,
        "isSeqModel": is_seq_model,
        "head": head,
        "inner": inner,
        "isl2Norm": is_l2_norm,
    }

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = None

    if isinstance(ckpt, dict) and "model" in ckpt and "sd" in ckpt["model"]:
        state_dict = ckpt["model"]["sd"]
        ckpt_args = ckpt["model"].get("args")
        if isinstance(ckpt_args, dict):
            model_args.update(ckpt_args)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported OCR checkpoint format: {checkpoint_path}")

    model = models.make({"name": "GPLPR", "args": model_args})
    model.load_state_dict(normalize_state_dict_keys(state_dict), strict=True)
    model = model.to(device)
    model.eval()
    return model, model_args


def decode_gplpr(logits: torch.Tensor, converter: strLabelConverter):
    pred_idx = logits.argmax(2)
    return converter.decode_list(pred_idx.detach().cpu())


def get_center_lr_for_ocr(lr: torch.Tensor):
    # lr: [B, C, H, W], C can be 3(center3) or 15(stack15)
    if lr.shape[1] == 15:
        center = lr[:, 6:9, :, :]
    else:
        center = lr
    return F.interpolate(center, size=(32, 96), mode="bicubic", align_corners=False)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img = img_tensor.detach().cpu().clamp(0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def sanitize_path_token(token: str) -> str:
    safe = []
    for ch in str(token):
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "track"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True, help="mf5 train config path")
    parser.add_argument("--sr-checkpoint", required=True, help="SR checkpoint (.pth)")
    parser.add_argument("--ocr-checkpoint", required=True, help="GPLPR checkpoint (.pth)")
    parser.add_argument("--output-dir", required=True, help="directory for metrics outputs")
    parser.add_argument("--save-sr-dir", default=None, help="optional directory to save per-track SR images")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--ocr-alphabet", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--ocr-nc", type=int, default=3)
    parser.add_argument("--ocr-k", type=int, default=7)
    parser.add_argument("--ocr-is-seq-model", type=str2bool, default=True)
    parser.add_argument("--ocr-head", type=int, default=2)
    parser.add_argument("--ocr-inner", type=int, default=256)
    parser.add_argument("--ocr-is-l2-norm", type=str2bool, default=True)
    args = parser.parse_args()

    with open(args.train_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_set = TrackSequenceDataset(
        data_root=cfg["data"]["root"],
        phase="validation",
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        seed=cfg.get("seed", 1996),
        scenario_filter=cfg["data"].get("scenario_filter"),
        layout_filter=cfg["data"].get("layout_filter"),
    )
    val_set = TrackSequenceWrapper(
        dataset=base_set,
        imgW=cfg["preprocess"]["imgW"],
        imgH=cfg["preprocess"]["imgH"],
        aug=False,
        image_aspect_ratio=cfg["preprocess"]["image_aspect_ratio"],
        background=cfg["preprocess"]["background"],
        phase="validation",
        input_mode=cfg["preprocess"].get("input_mode", "stack15"),
    )

    val_batch_size = args.batch_size or cfg["train"].get("val_batch_size", 32)
    loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=TrackSequenceWrapper.collate_fn,
    )

    sr_model, sr_args = load_sr_model(Path(args.sr_checkpoint), device)
    ocr_model, ocr_args = load_gplpr_model(
        checkpoint_path=Path(args.ocr_checkpoint),
        device=device,
        alphabet=args.ocr_alphabet,
        nc=args.ocr_nc,
        k=args.ocr_k,
        is_seq_model=args.ocr_is_seq_model,
        head=args.ocr_head,
        inner=args.ocr_inner,
        is_l2_norm=args.ocr_is_l2_norm,
    )

    converter = strLabelConverter(ocr_args.get("alphabet", args.ocr_alphabet))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / "val_ocr_per_sample.csv"
    summary_json = out_dir / "val_ocr_summary.json"
    save_sr_dir = Path(args.save_sr_dir) if args.save_sr_dir else None
    if save_sr_dir is not None:
        save_sr_dir.mkdir(parents=True, exist_ok=True)
    saved_sr_count = 0
    track_name_counter = {}

    total = 0
    exact_sr = 0
    exact_lr = 0
    ed_sum_sr = 0
    ed_sum_lr = 0
    char_total = 0

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val-ocr", leave=False):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            gt_texts = [clean_plate_text(t) for t in batch["gt"]]
            track_ids = batch["track_id"]

            sr = sr_model(lr)
            if isinstance(sr, tuple):
                sr = sr[0]

            _, sr_logits, _ = ocr_model(sr)
            sr_preds = [clean_plate_text(t) for t in decode_gplpr(sr_logits, converter)]

            lr_center = get_center_lr_for_ocr(lr)
            _, lr_logits, _ = ocr_model(lr_center)
            lr_preds = [clean_plate_text(t) for t in decode_gplpr(lr_logits, converter)]

            _ = hr  # keep for potential future extensions

            for idx, (track_id, gt, pred_sr, pred_lr) in enumerate(zip(track_ids, gt_texts, sr_preds, lr_preds)):
                ed_sr = edit_distance(pred_sr, gt)
                ed_lr = edit_distance(pred_lr, gt)
                exact_match_sr = int(pred_sr == gt)
                exact_match_lr = int(pred_lr == gt)
                sr_relpath = ""

                if save_sr_dir is not None:
                    base_name = sanitize_path_token(track_id)
                    seen = track_name_counter.get(base_name, 0)
                    track_name_counter[base_name] = seen + 1
                    if seen == 0:
                        dir_name = base_name
                    else:
                        dir_name = f"{base_name}__dup{seen:04d}"
                    track_dir = save_sr_dir / dir_name
                    track_dir.mkdir(parents=True, exist_ok=True)
                    sr_path = track_dir / "sr.png"
                    tensor_to_pil(sr[idx]).save(sr_path)
                    sr_relpath = str(sr_path.relative_to(save_sr_dir))
                    saved_sr_count += 1

                total += 1
                exact_sr += exact_match_sr
                exact_lr += exact_match_lr
                ed_sum_sr += ed_sr
                ed_sum_lr += ed_lr
                char_total += max(1, len(gt))

                rows.append(
                    {
                        "track_id": track_id,
                        "gt": gt,
                        "pred_sr": pred_sr,
                        "pred_lr_center": pred_lr,
                        "ed_sr": ed_sr,
                        "ed_lr_center": ed_lr,
                        "exact_sr": exact_match_sr,
                        "exact_lr_center": exact_match_lr,
                        "sr_relpath": sr_relpath,
                    }
                )

    summary = {
        "num_samples": total,
        "sr_exact_match_acc": (exact_sr / total) if total > 0 else 0.0,
        "lr_center_exact_match_acc": (exact_lr / total) if total > 0 else 0.0,
        "sr_avg_edit_distance": (ed_sum_sr / total) if total > 0 else 0.0,
        "lr_center_avg_edit_distance": (ed_sum_lr / total) if total > 0 else 0.0,
        "sr_cer": (ed_sum_sr / char_total) if char_total > 0 else 0.0,
        "lr_center_cer": (ed_sum_lr / char_total) if char_total > 0 else 0.0,
        "sr_checkpoint": str(Path(args.sr_checkpoint).resolve()),
        "ocr_checkpoint": str(Path(args.ocr_checkpoint).resolve()),
        "train_config": str(Path(args.train_config).resolve()),
        "sr_model_args": sr_args,
        "ocr_model_args": ocr_args,
        "saved_sr_count": saved_sr_count,
        "saved_sr_dir": str(save_sr_dir.resolve()) if save_sr_dir is not None else "",
    }

    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "track_id", "gt", "pred_sr", "pred_lr_center", "ed_sr", "ed_lr_center", "exact_sr", "exact_lr_center", "sr_relpath"
        ])
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved per-sample csv: {per_sample_csv}")
    print(f"saved summary json: {summary_json}")


if __name__ == "__main__":
    main()
