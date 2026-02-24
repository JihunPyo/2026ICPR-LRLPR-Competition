import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models


IMAGE_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
ANNOTATION_CANDIDATES = ("annotations.json", "annotation.json")
TRACK_DIR_RE = re.compile(r"^track[_-]\d+$", re.IGNORECASE)


def normalize_state_dict_keys(sd: dict):
    if not sd:
        return sd
    has_module_prefix = all(k.startswith("module.") for k in sd.keys())
    if not has_module_prefix:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def clean_plate_text(text: str) -> str:
    return str(text).replace("#", "").replace("-", "").replace(" ", "").strip()


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def parse_scenario_key(name: str) -> Optional[str]:
    lname = str(name).lower()
    patterns = [
        r"(?:scenario|senario)[-_]?([ab])",
        r"^([ab])$",
        r"[-_/]([ab])$",
    ]
    for pattern in patterns:
        m = re.search(pattern, lname)
        if m:
            return m.group(1).upper()
    return None


def is_track_dir_name(name: str) -> bool:
    return bool(TRACK_DIR_RE.match(name))


def find_annotation_file(track_dir: Path) -> Optional[Path]:
    for ann_name in ANNOTATION_CANDIDATES:
        p = track_dir / ann_name
        if p.exists():
            return p
    return None


def find_image_with_stem(track_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        p = track_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def resolve_effective_train_root(train_root: Path) -> Path:
    if not train_root.exists():
        raise FileNotFoundError(f"train_root does not exist: {train_root}")

    candidate_paths: List[Path] = [train_root]
    direct_train = train_root / "train"
    if direct_train.exists():
        candidate_paths.append(direct_train)

    for cand in candidate_paths:
        keys = {
            parse_scenario_key(p.name)
            for p in cand.iterdir()
            if p.is_dir() and parse_scenario_key(p.name) in {"A", "B"}
        }
        if keys:
            return cand

    found_scenario_dirs: List[Path] = []
    for parent_str, dir_names, _ in os.walk(train_root):
        for dname in dir_names:
            if parse_scenario_key(dname) in {"A", "B"}:
                found_scenario_dirs.append(Path(parent_str) / dname)

    if not found_scenario_dirs:
        raise RuntimeError(f"Failed to find Scenario-A/B directories under: {train_root}")

    parent_count: Dict[Path, int] = {}
    for p in found_scenario_dirs:
        parent_count[p.parent] = parent_count.get(p.parent, 0) + 1
    best_parent = sorted(parent_count.items(), key=lambda x: (len(x[0].parts), -x[1], x[0].as_posix()))[0][0]
    return best_parent


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


class StrLabelConverter:
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = "-" + alphabet

    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i, :]
            char_list = []
            for j in range(t_item.shape[0]):
                idx = int(t_item[j])
                if idx == 0:
                    continue
                char_list.append(self.alphabet[idx])
            texts.append("".join(char_list))
        return texts


@dataclass
class TrackRecord:
    scenario_name: str
    scenario_key: str
    layout_name: str
    track_name: str
    track_dir: Path
    gt_text: str
    hr_paths: List[str]
    lr_paths: List[str]

    @property
    def track_uid(self) -> str:
        if self.layout_name == ".":
            return f"{self.scenario_name}/{self.track_name}"
        return f"{self.scenario_name}/{self.layout_name}/{self.track_name}"


class HRImageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict]):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["hr_path"]).convert("RGB")
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(32, 96), mode="bicubic", align_corners=False).squeeze(0)
        return {
            "track_uid": row["track_uid"],
            "scenario_name": row["scenario_name"],
            "scenario_key": row["scenario_key"],
            "layout_name": row["layout_name"],
            "track_name": row["track_name"],
            "frame_idx": row["frame_idx"],
            "hr_path": row["hr_path"],
            "gt_text_raw": row["gt_text_raw"],
            "gt_text_norm": row["gt_text_norm"],
            "image": x,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "track_uid": [b["track_uid"] for b in batch],
            "scenario_name": [b["scenario_name"] for b in batch],
            "scenario_key": [b["scenario_key"] for b in batch],
            "layout_name": [b["layout_name"] for b in batch],
            "track_name": [b["track_name"] for b in batch],
            "frame_idx": [b["frame_idx"] for b in batch],
            "hr_path": [b["hr_path"] for b in batch],
            "gt_text_raw": [b["gt_text_raw"] for b in batch],
            "gt_text_norm": [b["gt_text_norm"] for b in batch],
            "image": torch.stack([b["image"] for b in batch], dim=0),
        }


def collect_tracks(
    effective_train_root: Path,
    strict_complete_hr5: bool,
) -> Tuple[List[TrackRecord], List[Dict]]:
    tracks: List[TrackRecord] = []
    skipped: List[Dict] = []

    scenario_dirs = sorted([p for p in effective_train_root.iterdir() if p.is_dir()])
    for scenario_dir in scenario_dirs:
        scenario_key = parse_scenario_key(scenario_dir.name)
        if scenario_key not in {"A", "B"}:
            continue

        for track_dir in sorted(scenario_dir.rglob("*")):
            if (not track_dir.is_dir()) or (not is_track_dir_name(track_dir.name)):
                continue

            layout_rel = track_dir.parent.relative_to(scenario_dir)
            layout_name = layout_rel.as_posix() if str(layout_rel) != "." else "."

            ann_path = find_annotation_file(track_dir)
            if ann_path is None:
                skipped.append(
                    {
                        "scenario_name": scenario_dir.name,
                        "scenario_key": scenario_key,
                        "layout_name": layout_name,
                        "track_name": track_dir.name,
                        "track_dir": str(track_dir),
                        "reason": "missing_annotation",
                    }
                )
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            gt_text = str(ann.get("plate_text", "")).strip()

            hr_paths: List[str] = []
            lr_paths: List[str] = []
            missing_stems: List[str] = []
            for i in range(1, 6):
                hr_p = find_image_with_stem(track_dir, f"hr-{i:03d}")
                lr_p = find_image_with_stem(track_dir, f"lr-{i:03d}")
                if hr_p is None:
                    missing_stems.append(f"hr-{i:03d}")
                if lr_p is None:
                    missing_stems.append(f"lr-{i:03d}")
                hr_paths.append("" if hr_p is None else str(hr_p))
                lr_paths.append("" if lr_p is None else str(lr_p))

            is_complete = all(bool(x) for x in hr_paths) and all(bool(x) for x in lr_paths)
            if strict_complete_hr5 and (not is_complete):
                skipped.append(
                    {
                        "scenario_name": scenario_dir.name,
                        "scenario_key": scenario_key,
                        "layout_name": layout_name,
                        "track_name": track_dir.name,
                        "track_dir": str(track_dir),
                        "reason": "incomplete_hr_or_lr_5",
                        "missing": ",".join(missing_stems),
                    }
                )
                continue

            tracks.append(
                TrackRecord(
                    scenario_name=scenario_dir.name,
                    scenario_key=scenario_key,
                    layout_name=layout_name,
                    track_name=track_dir.name,
                    track_dir=track_dir,
                    gt_text=gt_text,
                    hr_paths=hr_paths,
                    lr_paths=lr_paths,
                )
            )

    tracks = sorted(
        tracks,
        key=lambda x: (x.scenario_key, x.scenario_name, x.layout_name, x.track_name),
    )
    return tracks, skipped


def safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_args():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--train-root", default=None, help="Path to extracted train root.")
    src.add_argument("--input-zip", default=None, help="Path to raw-train-trainval.zip")

    parser.add_argument("--output-dir", required=True, help="Directory to save reports.")
    parser.add_argument("--temp-root", default=None, help="Temp directory for zip extraction.")
    parser.add_argument("--strict-complete-hr5", type=str2bool, default=True)
    parser.add_argument("--max-tracks", type=int, default=0, help="Debug cap. 0 means all tracks.")

    parser.add_argument("--ocr-checkpoint", required=True, help="GPLPR checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--ocr-alphabet", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--ocr-nc", type=int, default=3)
    parser.add_argument("--ocr-k", type=int, default=7)
    parser.add_argument("--ocr-is-seq-model", type=str2bool, default=True)
    parser.add_argument("--ocr-head", type=int, default=2)
    parser.add_argument("--ocr-inner", type=int, default=256)
    parser.add_argument("--ocr-is-l2-norm", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = build_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    temp_dir_to_clean: Optional[Path] = None
    try:
        if args.input_zip is not None:
            input_zip = Path(args.input_zip)
            if not input_zip.exists():
                raise FileNotFoundError(f"input zip not found: {input_zip}")
            temp_dir = tempfile.mkdtemp(prefix="raw_train_eval_", dir=args.temp_root)
            temp_dir_to_clean = Path(temp_dir)
            with zipfile.ZipFile(input_zip, "r") as zf:
                zf.extractall(temp_dir_to_clean)
            train_root = resolve_effective_train_root(temp_dir_to_clean)
        else:
            train_root = resolve_effective_train_root(Path(args.train_root))

        tracks, skipped_tracks = collect_tracks(
            effective_train_root=train_root,
            strict_complete_hr5=args.strict_complete_hr5,
        )
        if args.max_tracks > 0:
            tracks = tracks[: args.max_tracks]

        if len(tracks) == 0:
            raise RuntimeError("No tracks to evaluate after collection/filtering.")

        ocr_model, model_args = load_gplpr_model(
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
        converter = StrLabelConverter(model_args.get("alphabet", args.ocr_alphabet))

        image_rows_input: List[Dict] = []
        track_state: Dict[str, Dict] = {}
        for t in tracks:
            track_state[t.track_uid] = {
                "track_uid": t.track_uid,
                "scenario_name": t.scenario_name,
                "scenario_key": t.scenario_key,
                "layout_name": t.layout_name,
                "track_name": t.track_name,
                "gt_text_raw": t.gt_text,
                "gt_text_norm": clean_plate_text(t.gt_text),
                "lr_paths": t.lr_paths,
                "hr_paths": t.hr_paths,
                "num_images": 0,
                "num_correct": 0,
                "has_any_correct": 0,
                "best_correct_frame_idx": "",
                "best_correct_confidence": -1.0,
                "best_correct_hr_path": "",
            }

            for frame_idx in range(1, 6):
                hr_path = t.hr_paths[frame_idx - 1]
                if not hr_path:
                    continue
                image_rows_input.append(
                    {
                        "track_uid": t.track_uid,
                        "scenario_name": t.scenario_name,
                        "scenario_key": t.scenario_key,
                        "layout_name": t.layout_name,
                        "track_name": t.track_name,
                        "frame_idx": frame_idx,
                        "hr_path": hr_path,
                        "gt_text_raw": t.gt_text,
                        "gt_text_norm": clean_plate_text(t.gt_text),
                    }
                )

        dataset = HRImageDataset(image_rows_input)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=HRImageDataset.collate_fn,
        )

        per_image_rows: List[Dict] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="gplpr-hr5-ocr", leave=False):
                hr = batch["image"].to(device, non_blocking=True)
                _, logits, _ = ocr_model(hr)
                pred_idx = logits.argmax(2).detach().cpu()
                pred_texts = [clean_plate_text(x) for x in converter.decode_list(pred_idx)]
                probs = torch.softmax(logits, dim=2).detach().cpu()
                confs = probs.max(dim=2).values.mean(dim=1).tolist()

                for i in range(len(pred_texts)):
                    pred = pred_texts[i]
                    conf = float(confs[i])
                    gt_norm = batch["gt_text_norm"][i]
                    exact = int(pred == gt_norm)
                    frame_idx = int(batch["frame_idx"][i])
                    track_uid = batch["track_uid"][i]

                    row = {
                        "track_uid": track_uid,
                        "scenario_name": batch["scenario_name"][i],
                        "scenario_key": batch["scenario_key"][i],
                        "layout_name": batch["layout_name"][i],
                        "track_name": batch["track_name"][i],
                        "frame_idx": frame_idx,
                        "hr_path": batch["hr_path"][i],
                        "gt_text": batch["gt_text_raw"][i],
                        "gt_text_norm": gt_norm,
                        "pred_text": pred,
                        "confidence": conf,
                        "exact_match": exact,
                    }
                    per_image_rows.append(row)

                    st = track_state[track_uid]
                    st["num_images"] += 1
                    st["num_correct"] += exact
                    if exact:
                        st["has_any_correct"] = 1
                        best_conf = st["best_correct_confidence"]
                        best_idx = st["best_correct_frame_idx"]
                        should_update = (conf > best_conf) or (
                            conf == best_conf and (best_idx == "" or frame_idx < int(best_idx))
                        )
                        if should_update:
                            st["best_correct_confidence"] = conf
                            st["best_correct_frame_idx"] = frame_idx
                            st["best_correct_hr_path"] = batch["hr_path"][i]

        per_image_rows.sort(
            key=lambda x: (x["scenario_key"], x["scenario_name"], x["layout_name"], x["track_name"], x["frame_idx"])
        )

        per_track_rows: List[Dict] = []
        selected_rows: List[Dict] = []
        dropped_rows: List[Dict] = []

        for uid in sorted(track_state.keys()):
            st = track_state[uid]
            best_conf = (
                float(st["best_correct_confidence"])
                if st["best_correct_confidence"] >= 0.0
                else ""
            )
            per_track_rows.append(
                {
                    "track_uid": st["track_uid"],
                    "scenario_name": st["scenario_name"],
                    "scenario_key": st["scenario_key"],
                    "layout_name": st["layout_name"],
                    "track_name": st["track_name"],
                    "gt_text": st["gt_text_raw"],
                    "gt_text_norm": st["gt_text_norm"],
                    "num_images": st["num_images"],
                    "num_correct": st["num_correct"],
                    "has_any_correct": st["has_any_correct"],
                    "best_correct_frame_idx": st["best_correct_frame_idx"],
                    "best_correct_confidence": best_conf,
                    "best_correct_hr_path": st["best_correct_hr_path"],
                }
            )

            if st["has_any_correct"] == 1:
                selected_rows.append(
                    {
                        "track_uid": st["track_uid"],
                        "scenario_name": st["scenario_name"],
                        "scenario_key": st["scenario_key"],
                        "layout_name": st["layout_name"],
                        "track_name": st["track_name"],
                        "selected_frame_idx": st["best_correct_frame_idx"],
                        "selected_hr_path": st["best_correct_hr_path"],
                        "all_lr_paths": "|".join(st["lr_paths"]),
                        "gt_text": st["gt_text_raw"],
                        "gt_text_norm": st["gt_text_norm"],
                    }
                )
            else:
                dropped_rows.append(
                    {
                        "track_uid": st["track_uid"],
                        "scenario_name": st["scenario_name"],
                        "scenario_key": st["scenario_key"],
                        "layout_name": st["layout_name"],
                        "track_name": st["track_name"],
                        "gt_text": st["gt_text_raw"],
                        "gt_text_norm": st["gt_text_norm"],
                        "reason": "no_correct_hr_in_5",
                    }
                )

        scenario_counts = {"A": 0, "B": 0}
        scenario_hits = {"A": 0, "B": 0}
        for row in per_track_rows:
            sk = row["scenario_key"]
            if sk in {"A", "B"}:
                scenario_counts[sk] += 1
                scenario_hits[sk] += int(row["has_any_correct"])

        scenario_a_rate = safe_rate(scenario_hits["A"], scenario_counts["A"])
        scenario_b_rate = safe_rate(scenario_hits["B"], scenario_counts["B"])
        macro_rate = (scenario_a_rate + scenario_b_rate) / 2.0
        total_tracks = len(per_track_rows)
        total_hits = sum(int(x["has_any_correct"]) for x in per_track_rows)
        overall_rate = safe_rate(total_hits, total_tracks)

        scenario_summary_rows = [
            {
                "scope": "A",
                "num_tracks": scenario_counts["A"],
                "num_hit_tracks": scenario_hits["A"],
                "track_hit_rate": scenario_a_rate,
            },
            {
                "scope": "B",
                "num_tracks": scenario_counts["B"],
                "num_hit_tracks": scenario_hits["B"],
                "track_hit_rate": scenario_b_rate,
            },
            {
                "scope": "overall",
                "num_tracks": total_tracks,
                "num_hit_tracks": total_hits,
                "track_hit_rate": overall_rate,
            },
            {
                "scope": "macro_avg",
                "num_tracks": "",
                "num_hit_tracks": "",
                "track_hit_rate": macro_rate,
            },
        ]

        write_csv(
            out_dir / "per_image_ocr.csv",
            per_image_rows,
            [
                "track_uid",
                "scenario_name",
                "scenario_key",
                "layout_name",
                "track_name",
                "frame_idx",
                "hr_path",
                "gt_text",
                "gt_text_norm",
                "pred_text",
                "confidence",
                "exact_match",
            ],
        )
        write_csv(
            out_dir / "per_track_summary.csv",
            per_track_rows,
            [
                "track_uid",
                "scenario_name",
                "scenario_key",
                "layout_name",
                "track_name",
                "gt_text",
                "gt_text_norm",
                "num_images",
                "num_correct",
                "has_any_correct",
                "best_correct_frame_idx",
                "best_correct_confidence",
                "best_correct_hr_path",
            ],
        )
        write_csv(
            out_dir / "selected_tracks_manifest.csv",
            selected_rows,
            [
                "track_uid",
                "scenario_name",
                "scenario_key",
                "layout_name",
                "track_name",
                "selected_frame_idx",
                "selected_hr_path",
                "all_lr_paths",
                "gt_text",
                "gt_text_norm",
            ],
        )
        write_csv(
            out_dir / "dropped_tracks.csv",
            dropped_rows,
            [
                "track_uid",
                "scenario_name",
                "scenario_key",
                "layout_name",
                "track_name",
                "gt_text",
                "gt_text_norm",
                "reason",
            ],
        )
        write_csv(
            out_dir / "scenario_summary.csv",
            scenario_summary_rows,
            ["scope", "num_tracks", "num_hit_tracks", "track_hit_rate"],
        )
        if skipped_tracks:
            write_csv(
                out_dir / "skipped_tracks.csv",
                skipped_tracks,
                sorted({k for row in skipped_tracks for k in row.keys()}),
            )

        run_summary = {
            "train_root_effective": str(train_root),
            "ocr_checkpoint": str(Path(args.ocr_checkpoint).resolve()),
            "ocr_model_args": model_args,
            "strict_complete_hr5": bool(args.strict_complete_hr5),
            "max_tracks": int(args.max_tracks),
            "num_processed_images": len(per_image_rows),
            "num_processed_tracks": total_tracks,
            "num_selected_tracks": len(selected_rows),
            "num_dropped_tracks": len(dropped_rows),
            "num_skipped_tracks": len(skipped_tracks),
            "scenario_a_num_tracks": scenario_counts["A"],
            "scenario_b_num_tracks": scenario_counts["B"],
            "scenario_a_num_hit_tracks": scenario_hits["A"],
            "scenario_b_num_hit_tracks": scenario_hits["B"],
            "scenario_a_track_hit_rate": scenario_a_rate,
            "scenario_b_track_hit_rate": scenario_b_rate,
            "scenario_macro_avg_track_hit_rate": macro_rate,
            "overall_track_hit_rate": overall_rate,
            "overall_image_exact_match_rate": safe_rate(
                sum(int(x["exact_match"]) for x in per_image_rows),
                len(per_image_rows),
            ),
        }
        with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

        print(json.dumps(run_summary, indent=2, ensure_ascii=False))
        print(f"saved: {out_dir / 'per_image_ocr.csv'}")
        print(f"saved: {out_dir / 'per_track_summary.csv'}")
        print(f"saved: {out_dir / 'selected_tracks_manifest.csv'}")
        print(f"saved: {out_dir / 'dropped_tracks.csv'}")
        print(f"saved: {out_dir / 'scenario_summary.csv'}")
        print(f"saved: {out_dir / 'run_summary.json'}")
        if skipped_tracks:
            print(f"saved: {out_dir / 'skipped_tracks.csv'}")
    finally:
        if temp_dir_to_clean is not None and temp_dir_to_clean.exists():
            shutil.rmtree(temp_dir_to_clean, ignore_errors=True)


if __name__ == "__main__":
    main()
