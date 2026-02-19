# 2026-02-19
# Two-stage curriculum training for GPLPR using Scenario-A/B track 서브셋. 

import argparse
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from train_funcs.train_utils import strLabelConverter

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrackSample:
    scenario: str
    layout: str
    track_name: str
    track_dir: Path
    pairs: List[Tuple[str, str]]  # (hr_path, lr_path)

    @property
    def track_id(self) -> str:
        return f"{self.scenario}/{self.layout}/{self.track_name}"


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


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


def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    has_module_prefix = all(k.startswith("module.") for k in sd.keys())
    if not has_module_prefix:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def parse_scenario_key(name: str) -> Optional[str]:
    lname = name.lower()
    patterns = [
        r"(?:scenario|senario)[-_]?([ab])",
        r"^([ab])$",
        r"[-_/]([ab])$",
    ]
    for p in patterns:
        m = re.search(p, lname)
        if m:
            return m.group(1).upper()
    return None


TRACK_DIR_RE = re.compile(r"^track[_-]\d+$", re.IGNORECASE)
IMAGE_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
ANNOTATION_CANDIDATES = ("annotations.json", "annotation.json")


def is_track_dir_name(name: str) -> bool:
    return bool(TRACK_DIR_RE.match(name))


def find_annotation_file(track_dir: Path) -> Optional[Path]:
    for ann_name in ANNOTATION_CANDIDATES:
        ann_path = track_dir / ann_name
        if ann_path.exists():
            return ann_path
    return None


def find_image_with_stem(track_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        p = track_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _has_scenario_pair(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    keys = set()
    for child in path.iterdir():
        if not child.is_dir():
            continue
        key = parse_scenario_key(child.name)
        if key in {"A", "B"}:
            keys.add(key)
    return keys == {"A", "B"}


def resolve_effective_train_root(train_root: Path) -> Path:
    if not train_root.exists():
        raise FileNotFoundError(f"train_root does not exist: {train_root}")

    checked_paths: List[Path] = [train_root]
    if _has_scenario_pair(train_root):
        return train_root

    direct_train = train_root / "train"
    checked_paths.append(direct_train)
    if _has_scenario_pair(direct_train):
        return direct_train

    candidate_roots: List[Path] = []
    scenario_dirs: List[Path] = []
    for parent_str, dir_names, _ in os.walk(train_root):
        keys = set()
        for dname in dir_names:
            key = parse_scenario_key(dname)
            if key in {"A", "B"}:
                keys.add(key)
                scenario_dirs.append(Path(parent_str) / dname)
        if keys == {"A", "B"}:
            candidate_roots.append(Path(parent_str))

    if candidate_roots:
        candidate_roots = sorted(candidate_roots, key=lambda p: (len(p.parts), p.as_posix()))
        return candidate_roots[0]

    scenario_hints = sorted({p.as_posix() for p in scenario_dirs})[:20]
    checked_hint = [p.as_posix() for p in checked_paths]
    raise RuntimeError(
        "Unable to resolve effective train root with both Scenario-A/B. "
        f"checked_path={train_root.as_posix()} checked_candidates={checked_hint} "
        f"resolved_scenario_dirs={scenario_hints}"
    )


def collect_tracks(
    train_root: Path, all_frames: bool, frame_index: int
) -> Tuple[Dict[str, List[TrackSample]], Dict[str, Dict[str, int]]]:
    tracks_by_scenario: Dict[str, List[TrackSample]] = {"A": [], "B": []}
    stats: Dict[str, Dict[str, int]] = {
        "A": {
            "total_tracks": 0,
            "valid_tracks": 0,
            "missing_annotation": 0,
            "missing_frame_pairs": 0,
        },
        "B": {
            "total_tracks": 0,
            "valid_tracks": 0,
            "missing_annotation": 0,
            "missing_frame_pairs": 0,
        },
    }
    frame_indices = [1, 2, 3, 4, 5] if all_frames else [frame_index]

    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")

    scenario_dirs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    for scenario_dir in scenario_dirs:
        scenario_key = parse_scenario_key(scenario_dir.name)
        if scenario_key not in {"A", "B"}:
            continue

        layout_dirs = sorted([p for p in scenario_dir.iterdir() if p.is_dir()])
        for layout_dir in layout_dirs:
            if is_track_dir_name(layout_dir.name):
                track_dirs = [layout_dir]
                layout_name = "."
            else:
                track_dirs = sorted(
                    [p for p in layout_dir.iterdir() if p.is_dir() and is_track_dir_name(p.name)]
                )
                layout_name = layout_dir.name
            for track_dir in track_dirs:
                stats[scenario_key]["total_tracks"] += 1

                ann = find_annotation_file(track_dir)
                if ann is None:
                    stats[scenario_key]["missing_annotation"] += 1
                    continue

                pairs: List[Tuple[str, str]] = []
                for idx in frame_indices:
                    hr = find_image_with_stem(track_dir, f"hr-{idx:03d}")
                    lr = find_image_with_stem(track_dir, f"lr-{idx:03d}")
                    if hr is not None and lr is not None:
                        pairs.append((hr.as_posix(), lr.as_posix()))

                if not pairs:
                    stats[scenario_key]["missing_frame_pairs"] += 1
                    continue

                stats[scenario_key]["valid_tracks"] += 1
                tracks_by_scenario[scenario_key].append(
                    TrackSample(
                        scenario=scenario_dir.name,
                        layout=layout_name,
                        track_name=track_dir.name,
                        track_dir=track_dir,
                        pairs=pairs,
                    )
                )

    for scenario_key in ("A", "B"):
        s = stats[scenario_key]
        print(
            f"[collect:{scenario_key}] total_tracks={s['total_tracks']} "
            f"valid_tracks={s['valid_tracks']} "
            f"missing_annotation={s['missing_annotation']} "
            f"missing_frame_pairs={s['missing_frame_pairs']}"
        )

    return tracks_by_scenario, stats


def split_train_val_by_track(
    tracks: Sequence[TrackSample],
    val_ratio: float,
    seed: int,
) -> Tuple[List[TrackSample], List[TrackSample]]:
    if len(tracks) == 0:
        return [], []

    tracks = list(tracks)
    rng = random.Random(seed)
    rng.shuffle(tracks)

    if val_ratio <= 0.0:
        return tracks, []

    n_val = int(len(tracks) * val_ratio)
    if n_val <= 0 and len(tracks) >= 2:
        n_val = 1
    if n_val >= len(tracks):
        n_val = len(tracks) - 1

    val_tracks = tracks[:n_val]
    train_tracks = tracks[n_val:]
    return train_tracks, val_tracks


def make_split_lines(train_tracks: Sequence[TrackSample], val_tracks: Sequence[TrackSample]) -> List[str]:
    lines: List[str] = []
    for t in train_tracks:
        for hr, lr in t.pairs:
            lines.append(f"{hr};{lr};training")
    for t in val_tracks:
        for hr, lr in t.pairs:
            lines.append(f"{hr};{lr};validation")
    return lines


def write_split(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def make_loader_from_split(
    split_path: Path,
    phase: str,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
    background: str,
    preprocessed: bool,
    aug: bool,
) -> DataLoader:
    dataset_spec = {
        "name": "parallel_training",
        "args": {
            "path_split": split_path.as_posix(),
            "phase": phase,
        },
    }
    wrapper_spec = {
        "name": "parallel_images_lp",
        "args": {
            "imgW": img_w,
            "imgH": img_h,
            "aug": aug,
            "image_aspect_ratio": 3,
            "background": background,
            "preprocessed": preprocessed,
        },
    }
    dataset = datasets.make(dataset_spec)
    wrapper = datasets.make(wrapper_spec, args={"dataset": dataset})
    return DataLoader(
        wrapper,
        batch_size=batch_size,
        shuffle=(phase == "training"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=wrapper.collate_fn,
    )


def compute_batch_loss_and_preds(
    model: nn.Module,
    hr: torch.Tensor,
    gt_texts: Sequence[str],
    converter: strLabelConverter,
    criterion: nn.Module,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int, int, int]:
    _, logits, _ = model(hr.to(device, non_blocking=True))
    target = converter.encode_list(gt_texts, K=max_len).to(device, non_blocking=True)
    loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    pred_idx = logits.argmax(2).detach().cpu()
    pred_texts = converter.decode_list(pred_idx)
    pred_texts = [p.replace("-", "").strip() for p in pred_texts]
    gt_texts = [str(g).replace("-", "").replace(" ", "").strip() for g in gt_texts]
    correct = sum(int(p == g) for p, g in zip(pred_texts, gt_texts))

    total_ed = 0
    total_chars = 0
    for pred, gt in zip(pred_texts, gt_texts):
        total_ed += edit_distance(pred, gt)
        total_chars += max(1, len(gt))

    return loss, correct, len(gt_texts), total_ed, total_chars


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_args: dict,
    optimizer_name: str,
    optimizer_args: dict,
    epoch: int,
    best_acc: float,
    stage_name: str,
) -> None:
    payload = {
        "model": {"name": "GPLPR", "args": model_args, "sd": model.state_dict()},
        "optimizer": {"name": optimizer_name, "args": optimizer_args, "sd": optimizer.state_dict()},
        "epoch": epoch,
        "state": torch.get_rng_state(),
        "best_acc": best_acc,
        "stage": stage_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_weights_into_model(model: nn.Module, checkpoint_path: Path, strict: bool = False) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict) and "sd" in ckpt["model"]:
        state_dict = ckpt["model"]["sd"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = normalize_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"[WARN] missing keys when loading {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys when loading {checkpoint_path}: {len(unexpected)}")


def train_stage(
    stage_name: str,
    model: nn.Module,
    split_path: Path,
    output_dir: Path,
    device: torch.device,
    converter: strLabelConverter,
    model_args: dict,
    batch_size: int,
    num_workers: int,
    img_w: int,
    img_h: int,
    background: str,
    preprocessed: bool,
    epochs: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    lr_step: int,
    lr_gamma: float,
    wandb_run=None,
    global_epoch_offset: int = 0,
) -> Dict[str, str]:
    train_loader = make_loader_from_split(
        split_path=split_path,
        phase="training",
        batch_size=batch_size,
        num_workers=num_workers,
        img_w=img_w,
        img_h=img_h,
        background=background,
        preprocessed=preprocessed,
        aug=True,
    )
    val_loader = make_loader_from_split(
        split_path=split_path,
        phase="validation",
        batch_size=batch_size,
        num_workers=num_workers,
        img_w=img_w,
        img_h=img_h,
        background=background,
        preprocessed=preprocessed,
        aug=False,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    if n_train == 0:
        raise RuntimeError(f"No training samples found for {stage_name} in split: {split_path}")

    print(f"[{stage_name}] train samples={n_train} val samples={n_val}")

    criterion = nn.CrossEntropyLoss()
    optimizer_name_norm = optimizer_name.lower()
    optimizer_args = {"lr": lr, "weight_decay": weight_decay}
    if optimizer_name_norm == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    elif optimizer_name_norm == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    max_len = int(model_args.get("K", 7))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"
    summary_json = output_dir / "summary.json"
    best_ckpt = output_dir / f"best_{stage_name}.pth"
    last_ckpt = output_dir / f"last_{stage_name}.pth"

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "train_avg_ed",
                "train_cer",
                "val_loss",
                "val_acc",
                "val_avg_ed",
                "val_cer",
                "best_metric",
            ]
        )

    best_metric = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        train_ed_sum = 0
        train_char_sum = 0

        pbar = tqdm(train_loader, desc=f"{stage_name} train {epoch}/{epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss, correct, count, ed_sum, char_sum = compute_batch_loss_and_preds(
                model=model,
                hr=batch["hr"],
                gt_texts=batch["gt"],
                converter=converter,
                criterion=criterion,
                max_len=max_len,
                device=device,
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * count
            train_correct += correct
            train_count += count
            train_ed_sum += ed_sum
            train_char_sum += char_sum
            pbar.set_postfix(loss=f"{(train_loss_sum / max(train_count, 1)):.4f}")

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        train_avg_ed = train_ed_sum / max(train_count, 1)
        train_cer = train_ed_sum / max(train_char_sum, 1)

        val_loss = 0.0
        val_acc = 0.0
        val_avg_ed = 0.0
        val_cer = 0.0
        if n_val > 0:
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_count = 0
            val_ed_sum = 0
            val_char_sum = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"{stage_name} val {epoch}/{epochs}", leave=False):
                    loss, correct, count, ed_sum, char_sum = compute_batch_loss_and_preds(
                        model=model,
                        hr=batch["hr"],
                        gt_texts=batch["gt"],
                        converter=converter,
                        criterion=criterion,
                        max_len=max_len,
                        device=device,
                    )
                    val_loss_sum += loss.item() * count
                    val_correct += correct
                    val_count += count
                    val_ed_sum += ed_sum
                    val_char_sum += char_sum
            val_loss = val_loss_sum / max(val_count, 1)
            val_acc = val_correct / max(val_count, 1)
            val_avg_ed = val_ed_sum / max(val_count, 1)
            val_cer = val_ed_sum / max(val_char_sum, 1)

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        metric_now = val_acc if n_val > 0 else train_acc
        if metric_now > best_metric:
            best_metric = metric_now
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                model_args=model_args,
                optimizer_name=optimizer_name_norm,
                optimizer_args=optimizer_args,
                epoch=epoch,
                best_acc=best_metric,
                stage_name=stage_name,
            )

        save_checkpoint(
            path=last_ckpt,
            model=model,
            optimizer=optimizer,
            model_args=model_args,
            optimizer_name=optimizer_name_norm,
            optimizer_args=optimizer_args,
            epoch=epoch,
            best_acc=best_metric,
            stage_name=stage_name,
        )

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    lr_now,
                    train_loss,
                    train_acc,
                    train_avg_ed,
                    train_cer,
                    val_loss,
                    val_acc,
                    val_avg_ed,
                    val_cer,
                    best_metric,
                ]
            )

        summary = {
            "stage": stage_name,
            "last_epoch": epoch,
            "best_metric": best_metric,
            "train_samples": n_train,
            "val_samples": n_val,
            "latest": {
                "lr": lr_now,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_avg_ed": train_avg_ed,
                "train_cer": train_cer,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_avg_ed": val_avg_ed,
                "val_cer": val_cer,
            },
            "checkpoints": {
                "best": best_ckpt.as_posix(),
                "last": last_ckpt.as_posix(),
            },
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[{stage_name}] epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} best={best_metric:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "stage": stage_name,
                    "epoch": epoch,
                    "global_epoch": global_epoch_offset + epoch,
                    f"{stage_name}/lr": lr_now,
                    f"{stage_name}/train_loss": train_loss,
                    f"{stage_name}/train_acc": train_acc,
                    f"{stage_name}/train_avg_ed": train_avg_ed,
                    f"{stage_name}/train_cer": train_cer,
                    f"{stage_name}/val_loss": val_loss,
                    f"{stage_name}/val_acc": val_acc,
                    f"{stage_name}/val_avg_ed": val_avg_ed,
                    f"{stage_name}/val_cer": val_cer,
                    f"{stage_name}/best_metric": best_metric,
                }
            )

    return {
        "best_ckpt": best_ckpt.as_posix(),
        "last_ckpt": last_ckpt.as_posix(),
        "metrics_csv": metrics_csv.as_posix(),
        "summary_json": summary_json.as_posix(),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Two-stage curriculum training for GPLPR using Scenario-A/B track subsets."
    )
    p.add_argument(
        "--train-root",
        default="/data/pjh7639/datasets/raw_train_unzip",
        help="Raw train root. Supports either .../train or its parent directory.",
    )
    p.add_argument(
        "--pretrained-ckpt",
        default="/data/pjh7639/weights/GP_LPR/Rodosol.pth",
        help="Initial pretrained GPLPR checkpoint.",
    )
    p.add_argument(
        "--output-dir",
        default="./save/gplpr_curriculum",
        help="Output directory for splits, checkpoints, and logs.",
    )
    p.add_argument("--seed", type=int, default=1996)
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Per-stage validation split ratio by track.",
    )

    p.add_argument("--stage1-a-tracks", type=int, default=9000)
    p.add_argument("--stage1-b-tracks", type=int, default=4500)
    p.add_argument("--stage2-b-tracks", type=int, default=4500)

    p.add_argument("--all-frames", type=str2bool, default=True, help="Use all 5 frames per track.")
    p.add_argument("--frame-index", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Used if all_frames=false.")
    p.add_argument(
        "--dry-run-collect",
        type=str2bool,
        default=False,
        help="If true, only build/validate splits and exit before model training.",
    )

    p.add_argument("--img-w", type=int, default=48)
    p.add_argument("--img-h", type=int, default=16)
    p.add_argument("--background", default="(127, 127, 127)")
    p.add_argument("--preprocessed", type=str2bool, default=False)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--stage1-epochs", type=int, default=40)
    p.add_argument("--stage1-lr", type=float, default=1.0e-4)
    p.add_argument("--stage1-lr-step", type=int, default=10)
    p.add_argument("--stage1-lr-gamma", type=float, default=0.5)

    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--stage2-lr", type=float, default=5.0e-5)
    p.add_argument("--stage2-lr-step", type=int, default=10)
    p.add_argument("--stage2-lr-gamma", type=float, default=0.5)

    p.add_argument("--alphabet", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    p.add_argument("--nc", type=int, default=3)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--is-seq-model", type=str2bool, default=True)
    p.add_argument("--head", type=int, default=2)
    p.add_argument("--inner", type=int, default=256)
    p.add_argument("--is-l2-norm", type=str2bool, default=True)

    p.add_argument("--wandb-enabled", type=str2bool, default=False)
    p.add_argument("--wandb-project", default="gplpr-curriculum")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    requested_train_root = Path(args.train_root)
    resolved_train_root = resolve_effective_train_root(requested_train_root)
    if resolved_train_root != requested_train_root:
        print(
            "[INFO] auto-resolved train root: "
            f"requested={requested_train_root.as_posix()} resolved={resolved_train_root.as_posix()}"
        )
    else:
        print(f"[INFO] train root: {resolved_train_root.as_posix()}")

    tracks_by_scenario, collect_stats = collect_tracks(
        train_root=resolved_train_root,
        all_frames=args.all_frames,
        frame_index=args.frame_index,
    )
    tracks_a = tracks_by_scenario["A"]
    tracks_b = tracks_by_scenario["B"]
    print(f"[INFO] collected tracks: Scenario-A={len(tracks_a)} Scenario-B={len(tracks_b)}")

    rng = random.Random(args.seed)
    rng.shuffle(tracks_a)
    rng.shuffle(tracks_b)

    if len(tracks_a) < args.stage1_a_tracks:
        raise RuntimeError(f"Scenario-A tracks are not enough: have {len(tracks_a)}, need {args.stage1_a_tracks}")
    if len(tracks_b) < (args.stage1_b_tracks + args.stage2_b_tracks):
        need = args.stage1_b_tracks + args.stage2_b_tracks
        raise RuntimeError(f"Scenario-B tracks are not enough: have {len(tracks_b)}, need {need}")

    stage1_tracks = tracks_a[: args.stage1_a_tracks] + tracks_b[: args.stage1_b_tracks]
    stage2_tracks = tracks_b[args.stage1_b_tracks : args.stage1_b_tracks + args.stage2_b_tracks]

    s1_train_tracks, s1_val_tracks = split_train_val_by_track(
        tracks=stage1_tracks,
        val_ratio=args.val_ratio,
        seed=args.seed + 11,
    )
    s2_train_tracks, s2_val_tracks = split_train_val_by_track(
        tracks=stage2_tracks,
        val_ratio=args.val_ratio,
        seed=args.seed + 22,
    )

    stage1_split_path = splits_dir / "stage1_ab_split.txt"
    stage2_split_path = splits_dir / "stage2_b_split.txt"
    write_split(stage1_split_path, make_split_lines(s1_train_tracks, s1_val_tracks))
    write_split(stage2_split_path, make_split_lines(s2_train_tracks, s2_val_tracks))

    selection = {
        "seed": args.seed,
        "train_root_requested": requested_train_root.as_posix(),
        "train_root_resolved": resolved_train_root.as_posix(),
        "all_frames": bool(args.all_frames),
        "frame_index": int(args.frame_index),
        "collect_stats": collect_stats,
        "stage1": {
            "scenario_a_tracks": args.stage1_a_tracks,
            "scenario_b_tracks": args.stage1_b_tracks,
            "train_tracks": len(s1_train_tracks),
            "val_tracks": len(s1_val_tracks),
            "track_ids": [t.track_id for t in stage1_tracks],
            "split_file": stage1_split_path.as_posix(),
        },
        "stage2": {
            "scenario_b_tracks": args.stage2_b_tracks,
            "train_tracks": len(s2_train_tracks),
            "val_tracks": len(s2_val_tracks),
            "track_ids": [t.track_id for t in stage2_tracks],
            "split_file": stage2_split_path.as_posix(),
        },
    }
    with open(output_dir / "track_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    if args.dry_run_collect:
        print("[DONE] dry-run-collect enabled. Split generation and data checks completed.")
        print(f"[DONE] stage1 split: {stage1_split_path.as_posix()}")
        print(f"[DONE] stage2 split: {stage2_split_path.as_posix()}")
        return

    model_args = {
        "alphabet": args.alphabet,
        "nc": args.nc,
        "K": args.k,
        "isSeqModel": args.is_seq_model,
        "head": args.head,
        "inner": args.inner,
        "isl2Norm": args.is_l2_norm,
    }
    model = models.make({"name": "GPLPR", "args": model_args})
    load_weights_into_model(model, Path(args.pretrained_ckpt), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    converter = strLabelConverter(args.alphabet)

    wandb_run = None
    use_wandb = bool(args.wandb_enabled)
    if use_wandb and wandb is None:
        print("[WARN] wandb-enabled is true but wandb package is not installed. Continue without wandb.")
        use_wandb = False
    if use_wandb and args.wandb_mode == "disabled":
        use_wandb = False
    if use_wandb:
        wb_run_name = args.wandb_run_name or f"gplpr-curriculum-{output_dir.name}"
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wb_run_name,
                mode=args.wandb_mode,
                dir=args.wandb_dir,
                config={
                    "train_root_requested": requested_train_root.as_posix(),
                    "train_root_resolved": resolved_train_root.as_posix(),
                    "pretrained_ckpt": Path(args.pretrained_ckpt).as_posix(),
                    "output_dir": output_dir.as_posix(),
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "stage1_a_tracks": args.stage1_a_tracks,
                    "stage1_b_tracks": args.stage1_b_tracks,
                    "stage2_b_tracks": args.stage2_b_tracks,
                    "stage1_epochs": args.stage1_epochs,
                    "stage2_epochs": args.stage2_epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "img_w": args.img_w,
                    "img_h": args.img_h,
                    "background": args.background,
                },
            )
        except Exception as e:
            print(f"[WARN] wandb init failed. Continue without wandb: {e}")
            wandb_run = None

    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"

    print("[INFO] start stage1 training")
    stage1_result = train_stage(
        stage_name="stage1",
        model=model,
        split_path=stage1_split_path,
        output_dir=stage1_dir,
        device=device,
        converter=converter,
        model_args=model_args,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=args.img_w,
        img_h=args.img_h,
        background=args.background,
        preprocessed=args.preprocessed,
        epochs=args.stage1_epochs,
        optimizer_name=args.optimizer,
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
        lr_step=args.stage1_lr_step,
        lr_gamma=args.stage1_lr_gamma,
        wandb_run=wandb_run,
        global_epoch_offset=0,
    )

    stage1_best = Path(stage1_result["best_ckpt"])
    print(f"[INFO] load stage1 best checkpoint: {stage1_best}")
    load_weights_into_model(model, stage1_best, strict=True)

    print("[INFO] start stage2 training")
    stage2_result = train_stage(
        stage_name="stage2",
        model=model,
        split_path=stage2_split_path,
        output_dir=stage2_dir,
        device=device,
        converter=converter,
        model_args=model_args,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_w=args.img_w,
        img_h=args.img_h,
        background=args.background,
        preprocessed=args.preprocessed,
        epochs=args.stage2_epochs,
        optimizer_name=args.optimizer,
        lr=args.stage2_lr,
        weight_decay=args.weight_decay,
        lr_step=args.stage2_lr_step,
        lr_gamma=args.stage2_lr_gamma,
        wandb_run=wandb_run,
        global_epoch_offset=args.stage1_epochs,
    )

    run_summary = {
        "pretrained_ckpt": Path(args.pretrained_ckpt).as_posix(),
        "train_root_requested": requested_train_root.as_posix(),
        "train_root_resolved": resolved_train_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "stage1": stage1_result,
        "stage2": stage2_result,
        "track_selection": (output_dir / "track_selection.json").as_posix(),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    if wandb_run is not None:
        try:
            wandb_run.summary["stage1_best_ckpt"] = stage1_result["best_ckpt"]
            wandb_run.summary["stage2_best_ckpt"] = stage2_result["best_ckpt"]
            wandb_run.summary["run_summary"] = (output_dir / "run_summary.json").as_posix()
            wandb_run.finish()
        except Exception as e:
            print(f"[WARN] wandb finalize failed: {e}")

    print("[DONE] curriculum training finished.")
    print(f"[DONE] stage1 best: {stage1_result['best_ckpt']}")
    print(f"[DONE] stage2 best: {stage2_result['best_ckpt']}")


if __name__ == "__main__":
    main()
