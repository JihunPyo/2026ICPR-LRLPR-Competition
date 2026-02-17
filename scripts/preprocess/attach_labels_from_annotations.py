#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_line(line: str):
    parts = [x.strip() for x in line.strip().split(";")]
    if len(parts) != 3:
        raise ValueError(f"Invalid split line: {line}")
    return parts[0], parts[1], parts[2]


def load_lines(path: Path):
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Attach label txt files to preprocessed HR images using original annotations.json."
    )
    p.add_argument("--split-original", required=True, help="Original split txt (paths to raw train images)")
    p.add_argument("--split-preprocessed", required=True, help="Preprocessed split txt (paths to preprocessed images)")
    args = p.parse_args()

    orig_lines = load_lines(Path(args.split_original))
    prep_lines = load_lines(Path(args.split_preprocessed))

    if len(orig_lines) != len(prep_lines):
        raise RuntimeError(
            f"Line count mismatch: original={len(orig_lines)} preprocessed={len(prep_lines)}"
        )

    written = 0
    missing_ann = 0
    for orig, prep in zip(orig_lines, prep_lines):
        orig_hr, _, _ = parse_line(orig)
        prep_hr, _, _ = parse_line(prep)

        ann_path = Path(orig_hr).parent / "annotations.json"
        if not ann_path.exists():
            missing_ann += 1
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        plate = ann.get("plate_text", "")

        txt_path = Path(prep_hr).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(f"plate: {plate}\n", encoding="utf-8")
        written += 1

    print(f"[DONE] label txt written: {written}")
    print(f"[INFO] missing annotations: {missing_ann}")


if __name__ == "__main__":
    main()
