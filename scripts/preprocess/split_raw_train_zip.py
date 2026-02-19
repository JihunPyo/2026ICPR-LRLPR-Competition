#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TRACK_DIR_RE = re.compile(r"^track[_-]\d+$", re.IGNORECASE)
FRAME_RE = re.compile(r"^(lr|hr)-(\d{3})\.(jpg|jpeg|png)$", re.IGNORECASE)
ANNOTATION_CANDIDATES = {"annotations.json", "annotation.json"}


@dataclass
class TrackRecord:
    scenario_name: str
    scenario_key: str
    layout_name: str
    track_name: str
    track_id: str
    stratum_key: Tuple[str, str]
    source_entries: List[zipfile.ZipInfo] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    has_annotation: bool = False
    lr_indices: set = field(default_factory=set)
    hr_indices: set = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split raw-train zip by track into stratified val/test zips.",
    )
    p.add_argument(
        "--input-zip",
        default="/data/pjh7639/datasets/raw-train.zip",
        help="Input raw-train zip path",
    )
    p.add_argument(
        "--out-val-zip",
        default="/data/pjh7639/datasets/raw-train-trainval.zip",
        help="Output val zip path (90%% by default)",
    )
    p.add_argument(
        "--out-test-zip",
        default="/data/pjh7639/datasets/raw-train_test.zip",
        help="Output test zip path (10%% by default)",
    )
    p.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio by track")
    p.add_argument("--seed", type=int, default=1996, help="Random seed")
    p.add_argument(
        "--manifest-json",
        default=None,
        help="Manifest JSON output path (default: out-val-zip dir/raw-train_split_manifest.json)",
    )
    p.add_argument(
        "--val-track-list",
        default=None,
        help="Val track-id list output path (default: out-val-zip dir/raw-train_val_tracks.txt)",
    )
    p.add_argument(
        "--test-track-list",
        default=None,
        help="Test track-id list output path (default: out-val-zip dir/raw-train_test_tracks.txt)",
    )
    p.add_argument(
        "--overwrite",
        type=int,
        default=1,
        choices=[0, 1],
        help="Overwrite existing outputs (1: yes, 0: no)",
    )
    p.add_argument(
        "--strict",
        type=int,
        default=1,
        choices=[0, 1],
        help="Fail when invalid tracks are detected (1: fail, 0: skip invalid tracks)",
    )
    return p.parse_args()


def parse_scenario_key(name: str) -> Optional[str]:
    lname = name.lower()
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


def normalize_zip_path(path: str) -> str:
    p = path.replace("\\", "/").strip("/")
    if not p:
        return ""
    return "/".join(part for part in p.split("/") if part and part != ".")


def resolve_track_context(parts: Sequence[str]) -> Tuple[Optional[int], Optional[int]]:
    track_indices = [i for i, name in enumerate(parts) if is_track_dir_name(name)]
    if not track_indices:
        return None, None

    for track_idx in track_indices:
        for scenario_idx in range(track_idx - 1, -1, -1):
            if parse_scenario_key(parts[scenario_idx]) in {"A", "B"}:
                return scenario_idx, track_idx
    return None, track_indices[0]


def clone_info_with_new_name(info: zipfile.ZipInfo, new_name: str) -> zipfile.ZipInfo:
    zinfo = zipfile.ZipInfo(filename=new_name, date_time=info.date_time)
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
    if any(ord(ch) > 127 for ch in new_name):
        zinfo.flag_bits |= 0x800
    return zinfo


def validate_input_path(input_zip: Path) -> None:
    if input_zip.exists():
        return

    hints: List[str] = []
    if input_zip.name == "raw-trian.zip":
        hints.append("possible typo detected: raw-trian.zip -> raw-train.zip")
    if "raw-trian.zip" in input_zip.name:
        candidate = input_zip.with_name(input_zip.name.replace("raw-trian.zip", "raw-train.zip"))
        if candidate.exists():
            hints.append(f"existing candidate found: {candidate.as_posix()}")
    else:
        candidate = input_zip.with_name(input_zip.name.replace("raw-trian", "raw-train"))
        if candidate.exists() and candidate != input_zip:
            hints.append(f"did you mean: {candidate.as_posix()}")

    msg = f"Input zip not found: {input_zip.as_posix()}"
    if hints:
        msg += " | " + " | ".join(hints)
    raise FileNotFoundError(msg)


def resolve_outputs(
    out_val_zip: Path,
    out_test_zip: Path,
    manifest_json: Optional[str],
    val_track_list: Optional[str],
    test_track_list: Optional[str],
) -> Tuple[Path, Path, Path]:
    out_dir = out_val_zip.parent
    manifest_path = Path(manifest_json) if manifest_json else out_dir / "raw-train_split_manifest.json"
    val_list_path = Path(val_track_list) if val_track_list else out_dir / "raw-train_val_tracks.txt"
    test_list_path = Path(test_track_list) if test_track_list else out_dir / "raw-train_test_tracks.txt"
    return manifest_path, val_list_path, test_list_path


def prepare_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        exists = [p.as_posix() for p in paths if p.exists()]
        if exists:
            raise FileExistsError("Refusing to overwrite existing outputs:\n" + "\n".join(exists))


def make_temp_path(dst: Path) -> Path:
    return dst.with_name(f".{dst.name}.tmp.{os.getpid()}.{random.randint(1000, 9999)}")


def atomic_write_text(path: Path, content: str) -> None:
    tmp = make_temp_path(path)
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def scan_tracks_from_zip(
    zin: zipfile.ZipFile,
) -> Tuple[Dict[str, TrackRecord], Dict[str, int], List[dict], int]:
    tracks: Dict[str, TrackRecord] = {}
    stats = {
        "entries_total": 0,
        "entries_files": 0,
        "entries_track_files": 0,
        "entries_unparsed_track_files": 0,
        "invalid_no_scenario_tracks": 0,
        "invalid_missing_annotation_tracks": 0,
        "invalid_missing_frame_pair_tracks": 0,
    }
    invalid_examples: List[dict] = []
    unparsed_track_prefixes = set()

    for info in zin.infolist():
        stats["entries_total"] += 1
        if info.is_dir():
            continue
        stats["entries_files"] += 1

        norm_path = normalize_zip_path(info.filename)
        if not norm_path:
            continue
        parts = norm_path.split("/")

        scenario_idx, track_idx = resolve_track_context(parts)
        if track_idx is None:
            continue

        stats["entries_track_files"] += 1
        if scenario_idx is None:
            stats["entries_unparsed_track_files"] += 1
            unparsed_track_prefixes.add("/".join(parts[: track_idx + 1]))
            continue

        scenario_name = parts[scenario_idx]
        scenario_key = parse_scenario_key(scenario_name)
        if scenario_key not in {"A", "B"}:
            stats["entries_unparsed_track_files"] += 1
            unparsed_track_prefixes.add("/".join(parts[: track_idx + 1]))
            continue

        layout_parts = parts[scenario_idx + 1 : track_idx]
        layout_name = "/".join(layout_parts) if layout_parts else "."
        track_name = parts[track_idx]
        track_id = "/".join([scenario_name] + layout_parts + [track_name])

        suffix_parts = parts[track_idx + 1 :]
        output_name = "/".join(["train", scenario_name] + layout_parts + [track_name] + suffix_parts)

        rec = tracks.get(track_id)
        if rec is None:
            rec = TrackRecord(
                scenario_name=scenario_name,
                scenario_key=scenario_key,
                layout_name=layout_name,
                track_name=track_name,
                track_id=track_id,
                stratum_key=(scenario_key, layout_name),
            )
            tracks[track_id] = rec

        rec.source_entries.append(info)
        rec.output_names.append(output_name)

        base = parts[-1].lower()
        if base in ANNOTATION_CANDIDATES:
            rec.has_annotation = True

        frame_match = FRAME_RE.match(parts[-1])
        if frame_match:
            idx = int(frame_match.group(2))
            if frame_match.group(1).lower() == "lr":
                rec.lr_indices.add(idx)
            else:
                rec.hr_indices.add(idx)

    if unparsed_track_prefixes:
        for prefix in sorted(unparsed_track_prefixes)[:20]:
            invalid_examples.append({"reason": "no_scenario", "track_prefix": prefix})

    valid_tracks: Dict[str, TrackRecord] = {}
    for track_id, rec in tracks.items():
        if not rec.has_annotation:
            stats["invalid_missing_annotation_tracks"] += 1
            invalid_examples.append({"reason": "missing_annotation", "track_id": track_id})
            continue
        if not (rec.lr_indices & rec.hr_indices):
            stats["invalid_missing_frame_pair_tracks"] += 1
            invalid_examples.append({"reason": "missing_frame_pair", "track_id": track_id})
            continue
        valid_tracks[track_id] = rec

    stats["invalid_no_scenario_tracks"] = len(unparsed_track_prefixes)
    return valid_tracks, stats, invalid_examples[:50], len(tracks)


def allocate_by_largest_remainder(
    strata_sizes: Dict[Tuple[str, str], int],
    target: int,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    total = sum(strata_sizes.values())
    if total <= 0:
        raise ValueError("Cannot allocate on empty strata_sizes")
    if target < 0:
        raise ValueError("target must be non-negative")
    target = min(target, total)

    quotas: Dict[Tuple[str, str], float] = {}
    remainders: Dict[Tuple[str, str], float] = {}
    alloc: Dict[Tuple[str, str], int] = {}

    for key, size in strata_sizes.items():
        quota = target * (size / total)
        floor_q = int(math.floor(quota))
        quotas[key] = quota
        remainders[key] = quota - floor_q
        alloc[key] = floor_q

    remaining = target - sum(alloc.values())
    order = sorted(
        strata_sizes.keys(),
        key=lambda k: (-remainders[k], -strata_sizes[k], k[0], k[1]),
    )

    idx = 0
    while remaining > 0 and order:
        key = order[idx % len(order)]
        if alloc[key] < strata_sizes[key]:
            alloc[key] += 1
            remaining -= 1
        idx += 1
        if idx > len(order) * 8 and all(alloc[k] >= strata_sizes[k] for k in order):
            break

    if sum(alloc.values()) != target:
        raise RuntimeError(
            f"Internal allocation mismatch: expected target={target}, got={sum(alloc.values())}"
        )

    return alloc, quotas, remainders


def stratified_split(
    valid_tracks: Dict[str, TrackRecord],
    test_ratio: float,
    seed: int,
) -> Tuple[set, set, dict]:
    total_tracks = len(valid_tracks)
    if total_tracks <= 0:
        raise RuntimeError("No valid tracks found to split.")
    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError(f"test_ratio must be in [0, 1], got: {test_ratio}")

    target_test = max(1, int(total_tracks * test_ratio))
    target_test = min(target_test, total_tracks)

    strata_to_tracks: Dict[Tuple[str, str], List[str]] = {}
    for track_id, rec in valid_tracks.items():
        strata_to_tracks.setdefault(rec.stratum_key, []).append(track_id)

    strata_sizes = {k: len(v) for k, v in strata_to_tracks.items()}
    alloc, quotas, remainders = allocate_by_largest_remainder(strata_sizes, target_test)

    rng = random.Random(seed)
    test_tracks = set()
    strata_stats = []

    for key in sorted(strata_to_tracks.keys(), key=lambda x: (x[0], x[1])):
        scenario_key, layout_name = key
        track_ids = sorted(strata_to_tracks[key])
        rng.shuffle(track_ids)
        n_test = alloc[key]
        chosen = track_ids[:n_test]
        test_tracks.update(chosen)

        strata_stats.append(
            {
                "scenario_key": scenario_key,
                "layout_name": layout_name,
                "total_tracks": len(track_ids),
                "test_tracks": n_test,
                "val_tracks": len(track_ids) - n_test,
                "quota": quotas[key],
                "remainder": remainders[key],
            }
        )

    all_tracks = set(valid_tracks.keys())
    val_tracks = all_tracks - test_tracks

    if test_tracks & val_tracks:
        raise RuntimeError("Split overlap detected between test and val tracks")
    if len(test_tracks) != target_test:
        raise RuntimeError(
            f"Split size mismatch: expected test={target_test}, got={len(test_tracks)}"
        )
    if len(test_tracks) + len(val_tracks) != total_tracks:
        raise RuntimeError("Split does not cover all valid tracks")

    split_summary = {
        "total_tracks": total_tracks,
        "target_test_tracks": target_test,
        "actual_test_tracks": len(test_tracks),
        "actual_val_tracks": len(val_tracks),
    }
    return val_tracks, test_tracks, {"summary": split_summary, "strata": strata_stats}


def write_split_zips(
    zin: zipfile.ZipFile,
    valid_tracks: Dict[str, TrackRecord],
    val_tracks: set,
    test_tracks: set,
    out_val_zip: Path,
    out_test_zip: Path,
) -> dict:
    tmp_val = make_temp_path(out_val_zip)
    tmp_test = make_temp_path(out_test_zip)
    written = {
        "val_files": 0,
        "test_files": 0,
        "val_bytes": 0,
        "test_bytes": 0,
    }
    seen_val = set()
    seen_test = set()

    try:
        with zipfile.ZipFile(tmp_val, mode="w", allowZip64=True) as zout_val, zipfile.ZipFile(
            tmp_test,
            mode="w",
            allowZip64=True,
        ) as zout_test:
            for track_id, rec in sorted(valid_tracks.items(), key=lambda x: x[0]):
                if track_id in test_tracks:
                    zout = zout_test
                    seen = seen_test
                    file_count_key = "test_files"
                    byte_count_key = "test_bytes"
                elif track_id in val_tracks:
                    zout = zout_val
                    seen = seen_val
                    file_count_key = "val_files"
                    byte_count_key = "val_bytes"
                else:
                    raise RuntimeError(f"Track not present in val/test split: {track_id}")

                for src_info, out_name in zip(rec.source_entries, rec.output_names):
                    if not out_name.startswith("train/"):
                        raise RuntimeError(f"Output path must start with train/: {out_name}")
                    if out_name in seen:
                        raise RuntimeError(f"Duplicate output path detected: {out_name}")
                    seen.add(out_name)

                    out_info = clone_info_with_new_name(src_info, out_name)
                    with zin.open(src_info, "r") as src_fp, zout.open(out_info, "w") as dst_fp:
                        shutil.copyfileobj(src_fp, dst_fp, length=1024 * 1024)

                    written[file_count_key] += 1
                    written[byte_count_key] += int(src_info.file_size)

        os.replace(tmp_val, out_val_zip)
        os.replace(tmp_test, out_test_zip)
    finally:
        if tmp_val.exists():
            tmp_val.unlink(missing_ok=True)
        if tmp_test.exists():
            tmp_test.unlink(missing_ok=True)

    return written


def main() -> int:
    args = parse_args()

    input_zip = Path(args.input_zip)
    out_val_zip = Path(args.out_val_zip)
    out_test_zip = Path(args.out_test_zip)
    overwrite = bool(args.overwrite)
    strict = bool(args.strict)

    if out_val_zip == out_test_zip:
        raise ValueError("--out-val-zip and --out-test-zip must be different paths")
    if input_zip == out_val_zip or input_zip == out_test_zip:
        raise ValueError("Output zip path must differ from input zip path")

    validate_input_path(input_zip)

    manifest_path, val_track_list_path, test_track_list_path = resolve_outputs(
        out_val_zip=out_val_zip,
        out_test_zip=out_test_zip,
        manifest_json=args.manifest_json,
        val_track_list=args.val_track_list,
        test_track_list=args.test_track_list,
    )

    prepare_output_paths(
        [out_val_zip, out_test_zip, manifest_path, val_track_list_path, test_track_list_path],
        overwrite=overwrite,
    )

    with zipfile.ZipFile(input_zip, mode="r") as zin:
        valid_tracks, scan_stats, invalid_examples, discovered_tracks = scan_tracks_from_zip(zin)

        invalid_total = (
            scan_stats["invalid_no_scenario_tracks"]
            + scan_stats["invalid_missing_annotation_tracks"]
            + scan_stats["invalid_missing_frame_pair_tracks"]
        )
        if strict and invalid_total > 0:
            sample = "\n".join(json.dumps(x, ensure_ascii=False) for x in invalid_examples[:20])
            raise RuntimeError(
                "Invalid tracks detected in strict mode.\n"
                f"- invalid_no_scenario_tracks={scan_stats['invalid_no_scenario_tracks']}\n"
                f"- invalid_missing_annotation_tracks={scan_stats['invalid_missing_annotation_tracks']}\n"
                f"- invalid_missing_frame_pair_tracks={scan_stats['invalid_missing_frame_pair_tracks']}\n"
                "Examples:\n"
                f"{sample}"
            )

        val_tracks, test_tracks, split_stats = stratified_split(
            valid_tracks=valid_tracks,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        written_stats = write_split_zips(
            zin=zin,
            valid_tracks=valid_tracks,
            val_tracks=val_tracks,
            test_tracks=test_tracks,
            out_val_zip=out_val_zip,
            out_test_zip=out_test_zip,
        )

    atomic_write_text(val_track_list_path, "\n".join(sorted(val_tracks)) + "\n")
    atomic_write_text(test_track_list_path, "\n".join(sorted(test_tracks)) + "\n")

    manifest = {
        "input_zip": input_zip.as_posix(),
        "output": {
            "val_zip": out_val_zip.as_posix(),
            "test_zip": out_test_zip.as_posix(),
            "manifest_json": manifest_path.as_posix(),
            "val_track_list": val_track_list_path.as_posix(),
            "test_track_list": test_track_list_path.as_posix(),
        },
        "params": {
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "overwrite": int(overwrite),
            "strict": int(strict),
        },
        "scan_stats": {
            **scan_stats,
            "discovered_tracks": discovered_tracks,
            "valid_tracks": len(val_tracks) + len(test_tracks),
            "invalid_total": (
                scan_stats["invalid_no_scenario_tracks"]
                + scan_stats["invalid_missing_annotation_tracks"]
                + scan_stats["invalid_missing_frame_pair_tracks"]
            ),
        },
        "split": split_stats,
        "written": written_stats,
        "invalid_examples": invalid_examples,
    }
    atomic_write_json(manifest_path, manifest)

    print("[DONE] split completed")
    print(f"[INFO] input zip: {input_zip.as_posix()}")
    print(f"[INFO] out val zip: {out_val_zip.as_posix()}")
    print(f"[INFO] out test zip: {out_test_zip.as_posix()}")
    print(f"[INFO] manifest: {manifest_path.as_posix()}")
    print(
        f"[INFO] tracks total={manifest['split']['summary']['total_tracks']} "
        f"val={manifest['split']['summary']['actual_val_tracks']} "
        f"test={manifest['split']['summary']['actual_test_tracks']}"
    )
    print(
        f"[INFO] files written val={written_stats['val_files']} test={written_stats['test_files']} "
        f"bytes val={written_stats['val_bytes']} test={written_stats['test_bytes']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
