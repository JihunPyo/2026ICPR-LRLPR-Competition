#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


TRACK_DIR_RE = re.compile(r"^track[_-]\d+$", re.IGNORECASE)
HR_FILE_RE = re.compile(r"^hr-(\d{3})\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class TrackKey:
    scenario: str
    layout: str
    track: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build /data/pjh7639/datasets/selected-train-trainval.zip from manifest. "
            "Keeps selected tracks only, and keeps a single selected HR image per track."
        )
    )
    p.add_argument(
        "--manifest-csv",
        default="/data/pjh7639/weights/GP_LPR/evals/Tuned_gplpr_hr5_eval_20260223_182841/selected_tracks_manifest.csv",
        help="Path to selected_tracks_manifest.csv",
    )
    p.add_argument(
        "--input-zip",
        default="/data/pjh7639/datasets/raw-train-trainval.zip",
        help="Path to raw-train-trainval.zip",
    )
    p.add_argument(
        "--output-zip",
        default="/data/pjh7639/datasets/selected-train-trainval.zip",
        help="Path to write selected zip",
    )
    p.add_argument(
        "--strict-manifest-coverage",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, fail when some manifest tracks are not found in input zip.",
    )
    p.add_argument(
        "--overwrite",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, overwrite output zip.",
    )
    return p.parse_args()


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


def normalize_zip_path(path: str) -> str:
    p = path.replace("\\", "/").strip("/")
    return "/".join(x for x in p.split("/") if x and x != ".")


def clone_info(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    z = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    z.comment = info.comment
    z.extra = info.extra
    z.create_system = info.create_system
    z.create_version = info.create_version
    z.extract_version = info.extract_version
    z.reserved = info.reserved
    z.flag_bits = info.flag_bits
    z.volume = info.volume
    z.internal_attr = info.internal_attr
    z.external_attr = info.external_attr
    z.compress_type = info.compress_type
    return z


def parse_track_context(parts: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    track_idx = None
    for i, part in enumerate(parts):
        if TRACK_DIR_RE.match(part):
            track_idx = i
            break
    if track_idx is None:
        return None, None, None, None

    scenario_idx = None
    for i in range(track_idx - 1, -1, -1):
        if parse_scenario_key(parts[i]) in {"A", "B"}:
            scenario_idx = i
            break
    if scenario_idx is None:
        return None, None, None, None

    scenario_name = parts[scenario_idx]
    scenario_key = parse_scenario_key(scenario_name) or ""
    layout_name = "/".join(parts[scenario_idx + 1 : track_idx]) if (track_idx - scenario_idx > 1) else "."
    track_name = parts[track_idx]
    return scenario_name, scenario_key, layout_name, track_name


def load_manifest(manifest_csv: Path) -> Tuple[Dict[TrackKey, int], Dict[TrackKey, int]]:
    by_full: Dict[TrackKey, int] = {}
    by_norm: Dict[TrackKey, int] = {}

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = {"scenario_name", "scenario_key", "layout_name", "track_name", "selected_frame_idx"}
        miss = [c for c in need if c not in (reader.fieldnames or [])]
        if miss:
            raise ValueError(f"manifest missing columns: {miss}")

        for row in reader:
            scenario_name = (row.get("scenario_name") or "").strip()
            scenario_key = (row.get("scenario_key") or "").strip().upper()
            layout_name = (row.get("layout_name") or ".").strip() or "."
            track_name = (row.get("track_name") or "").strip()
            frame_raw = (row.get("selected_frame_idx") or "").strip()

            if not scenario_name or not track_name or not frame_raw:
                continue

            frame_idx = int(frame_raw)
            if frame_idx < 1 or frame_idx > 5:
                raise ValueError(f"selected_frame_idx out of range [1,5]: {frame_idx}")

            full_key = TrackKey(scenario_name, layout_name, track_name)
            norm_key = TrackKey(scenario_key if scenario_key else (parse_scenario_key(scenario_name) or ""), layout_name, track_name)

            old = by_full.get(full_key)
            if old is not None and old != frame_idx:
                raise ValueError(f"conflicting frame idx for {full_key}: {old} vs {frame_idx}")
            by_full[full_key] = frame_idx

            old_n = by_norm.get(norm_key)
            if old_n is not None and old_n != frame_idx:
                raise ValueError(f"conflicting frame idx for {norm_key}: {old_n} vs {frame_idx}")
            by_norm[norm_key] = frame_idx

    if not by_full:
        raise RuntimeError("No valid rows in manifest.")
    return by_full, by_norm


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    input_zip = Path(args.input_zip)
    output_zip = Path(args.output_zip)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_csv}")
    if not input_zip.exists():
        raise FileNotFoundError(f"input zip not found: {input_zip}")
    if output_zip.exists() and args.overwrite != 1:
        raise FileExistsError(f"output exists and overwrite=0: {output_zip}")

    by_full, by_norm = load_manifest(manifest_csv)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = output_zip.with_name(f".{output_zip.name}.tmp.{os.getpid()}")
    if tmp_zip.exists():
        tmp_zip.unlink()

    kept_tracks = set()
    kept_files = kept_hr = kept_lr = kept_ann = scanned_files = 0

    try:
        with zipfile.ZipFile(input_zip, "r") as zin, zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zout:
            for info in zin.infolist():
                if info.is_dir():
                    continue
                scanned_files += 1

                norm = normalize_zip_path(info.filename)
                if not norm:
                    continue
                parts = norm.split("/")

                scenario_name, scenario_key, layout_name, track_name = parse_track_context(parts)
                if not scenario_name:
                    continue

                key_full = TrackKey(scenario_name, layout_name, track_name)
                selected_idx = by_full.get(key_full)
                if selected_idx is None:
                    key_norm = TrackKey(scenario_key, layout_name, track_name)
                    selected_idx = by_norm.get(key_norm)
                if selected_idx is None:
                    continue

                base = parts[-1]
                hr_match = HR_FILE_RE.match(base)
                if hr_match is not None:
                    hr_idx = int(hr_match.group(1))
                    if hr_idx != selected_idx:
                        continue

                zout.writestr(clone_info(info), zin.read(info))
                kept_tracks.add(key_full)
                kept_files += 1

                lb = base.lower()
                if lb.startswith("lr-"):
                    kept_lr += 1
                elif lb.startswith("hr-"):
                    kept_hr += 1
                elif lb in {"annotations.json", "annotation.json"}:
                    kept_ann += 1

        os.replace(tmp_zip, output_zip)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)

    manifest_tracks = len(by_full)
    matched_tracks = len(kept_tracks)
    missing_tracks = manifest_tracks - matched_tracks

    print(f"manifest_csv: {manifest_csv}")
    print(f"input_zip: {input_zip}")
    print(f"output_zip: {output_zip}")
    print(f"manifest_tracks: {manifest_tracks}")
    print(f"matched_tracks: {matched_tracks}")
    print(f"missing_manifest_tracks_in_zip: {missing_tracks}")
    print(f"scanned_files: {scanned_files}")
    print(f"kept_files: {kept_files}")
    print(f"kept_lr_files: {kept_lr}")
    print(f"kept_hr_files: {kept_hr}")
    print(f"kept_annotation_files: {kept_ann}")
    print(f"output_zip_size_bytes: {output_zip.stat().st_size}")

    if args.strict_manifest_coverage == 1 and missing_tracks > 0:
        print(
            "[ERROR] strict_manifest_coverage=1 and some manifest tracks were not found in zip.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
