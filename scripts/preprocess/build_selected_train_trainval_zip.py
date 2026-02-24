#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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
            "Build selected-train-trainval.zip from raw-train-trainval.zip "
            "using selected_tracks_manifest.csv. "
            "For each selected track, keep all LR files and only one selected HR frame."
        )
    )
    p.add_argument(
        "--manifest-csv",
        required=True,
        help="Path to selected_tracks_manifest.csv",
    )
    p.add_argument(
        "--input-zip",
        default="/data/pjh7639/datasets/raw-train-trainval.zip",
        help="Input raw-train-trainval zip path",
    )
    p.add_argument(
        "--output-zip",
        default="/data/pjh7639/datasets/selected-train-trainval.zip",
        help="Output selected zip path",
    )
    p.add_argument(
        "--strict-manifest-coverage",
        type=int,
        default=1,
        choices=[0, 1],
        help="Fail if any manifest track is not found in input zip.",
    )
    p.add_argument(
        "--overwrite",
        type=int,
        default=1,
        choices=[0, 1],
        help="Overwrite output zip if exists.",
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
    if not p:
        return ""
    return "/".join(x for x in p.split("/") if x and x != ".")


def clone_info_with_same_name(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    zinfo = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    zinfo.comment = info.comment
    zinfo.extra = info.extra
    zinfo.create_system = info.create_system
    zinfo.create_version = info.create_version
    zinfo.extract_version = info.extract_version
    zinfo.reserved = info.reserved
    zinfo.flag_bits = info.flag_bits
    zinfo.volume = info.volume
    zinfo.internal_attr = info.internal_attr
    zinfo.external_attr = info.external_attr
    zinfo.compress_type = info.compress_type
    return zinfo


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
    layout_parts = parts[scenario_idx + 1 : track_idx]
    layout_name = "/".join(layout_parts) if layout_parts else "."
    track_name = parts[track_idx]
    return scenario_name, scenario_key, layout_name, track_name


def load_manifest(manifest_csv: Path) -> Tuple[Dict[TrackKey, int], Dict[TrackKey, int]]:
    by_full: Dict[TrackKey, int] = {}
    by_norm: Dict[TrackKey, int] = {}

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"scenario_name", "scenario_key", "layout_name", "track_name", "selected_frame_idx"}
        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"manifest missing required columns: {missing_cols}")

        for row in reader:
            scenario_name = (row.get("scenario_name") or "").strip()
            scenario_key = (row.get("scenario_key") or "").strip().upper()
            layout_name = (row.get("layout_name") or ".").strip() or "."
            track_name = (row.get("track_name") or "").strip()
            idx_raw = (row.get("selected_frame_idx") or "").strip()

            if not scenario_name or not track_name or not idx_raw:
                continue

            try:
                frame_idx = int(idx_raw)
            except ValueError:
                raise ValueError(f"invalid selected_frame_idx: {idx_raw}")
            if frame_idx < 1 or frame_idx > 5:
                raise ValueError(f"selected_frame_idx out of range [1,5]: {frame_idx}")

            full_key = TrackKey(scenario=scenario_name, layout=layout_name, track=track_name)
            norm_key = TrackKey(
                scenario=scenario_key if scenario_key else (parse_scenario_key(scenario_name) or ""),
                layout=layout_name,
                track=track_name,
            )

            old_full = by_full.get(full_key)
            if old_full is not None and old_full != frame_idx:
                raise ValueError(
                    f"conflicting selected_frame_idx for {full_key}: {old_full} vs {frame_idx}"
                )
            by_full[full_key] = frame_idx

            old_norm = by_norm.get(norm_key)
            if old_norm is not None and old_norm != frame_idx:
                raise ValueError(
                    f"conflicting selected_frame_idx for {norm_key}: {old_norm} vs {frame_idx}"
                )
            by_norm[norm_key] = frame_idx

    if not by_full:
        raise RuntimeError("manifest has no valid rows.")

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
        raise FileExistsError(f"output zip exists and overwrite=0: {output_zip}")

    by_full, by_norm = load_manifest(manifest_csv)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = output_zip.with_name(f".{output_zip.name}.tmp.{os.getpid()}")
    if tmp_zip.exists():
        tmp_zip.unlink()

    kept_files = 0
    kept_lr = 0
    kept_hr = 0
    kept_annotations = 0
    scanned_files = 0
    kept_track_keys: set = set()
    discovered_track_keys: set = set()

    try:
        with zipfile.ZipFile(input_zip, "r") as zin, zipfile.ZipFile(
            tmp_zip, "w", allowZip64=True
        ) as zout:
            for info in zin.infolist():
                if info.is_dir():
                    continue

                scanned_files += 1
                norm_path = normalize_zip_path(info.filename)
                if not norm_path:
                    continue
                parts = norm_path.split("/")

                scenario_name, scenario_key, layout_name, track_name = parse_track_context(parts)
                if not scenario_name:
                    continue

                key_full = TrackKey(scenario=scenario_name, layout=layout_name, track=track_name)
                discovered_track_keys.add(key_full)
                selected_idx = by_full.get(key_full)
                if selected_idx is None:
                    key_norm = TrackKey(scenario=scenario_key, layout=layout_name, track=track_name)
                    selected_idx = by_norm.get(key_norm)
                if selected_idx is None:
                    continue

                base_name = parts[-1]
                hr_match = HR_FILE_RE.match(base_name)
                if hr_match is not None:
                    hr_idx = int(hr_match.group(1))
                    if hr_idx != selected_idx:
                        continue

                data = zin.read(info)
                zout.writestr(clone_info_with_same_name(info), data)
                kept_files += 1
                kept_track_keys.add(key_full)

                lname = base_name.lower()
                if lname.startswith("lr-"):
                    kept_lr += 1
                elif lname.startswith("hr-"):
                    kept_hr += 1
                elif lname in {"annotations.json", "annotation.json"}:
                    kept_annotations += 1

        os.replace(tmp_zip, output_zip)
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)

    manifest_track_count = len(by_full)
    matched_track_count = len(kept_track_keys)
    missing_from_zip = manifest_track_count - matched_track_count

    print(f"manifest_csv: {manifest_csv}")
    print(f"input_zip: {input_zip}")
    print(f"output_zip: {output_zip}")
    print(f"manifest_tracks: {manifest_track_count}")
    print(f"matched_tracks: {matched_track_count}")
    print(f"missing_manifest_tracks_in_zip: {missing_from_zip}")
    print(f"scanned_files: {scanned_files}")
    print(f"kept_files: {kept_files}")
    print(f"kept_lr_files: {kept_lr}")
    print(f"kept_hr_files: {kept_hr}")
    print(f"kept_annotation_files: {kept_annotations}")
    print(f"output_zip_size_bytes: {output_zip.stat().st_size}")

    if args.strict_manifest_coverage == 1 and missing_from_zip > 0:
        print(
            "[ERROR] strict_manifest_coverage=1 and some manifest tracks were not found in input zip.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
