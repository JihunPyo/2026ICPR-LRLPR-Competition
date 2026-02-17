#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Ensure project root is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.preprocess import parse_background_color, resize_with_aspect_and_gray_padding


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess parallel split images (LR/HR) and write a new split file."
    )
    p.add_argument("--split-in", required=True, help="input split txt (hr;lr;split)")
    p.add_argument("--out-root", required=True, help="output root directory")
    p.add_argument("--split-out", required=True, help="output split txt path")
    p.add_argument("--lr-w", type=int, default=48)
    p.add_argument("--lr-h", type=int, default=16)
    p.add_argument("--background", default="(127, 127, 127)")
    p.add_argument("--ext", default=".png", help="output image extension")
    return p.parse_args()


def read_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def uid_for_pair(hr_path: str, lr_path: str):
    return hashlib.sha1(f"{hr_path}|{lr_path}".encode("utf-8")).hexdigest()[:16]


def main():
    args = parse_args()
    split_in = Path(args.split_in)
    out_root = Path(args.out_root)
    split_out = Path(args.split_out)
    bg = parse_background_color(args.background)

    lines = [line.strip() for line in split_in.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_lines = []
    missing_txt = 0
    missing_annotations = 0

    for line in tqdm(lines, desc="preprocess"):
        parts = [x.strip() for x in line.split(";")]
        if len(parts) != 3:
            raise ValueError(f"Invalid split line: {line}")
        hr_src, lr_src, split = parts

        hr_src_p = Path(hr_src)
        lr_src_p = Path(lr_src)
        uid = uid_for_pair(hr_src, lr_src)

        lr_dst = out_root / "lr" / split / f"{uid}{args.ext}"
        hr_dst = out_root / "hr" / split / f"{uid}{args.ext}"
        hr_txt_dst = hr_dst.with_suffix(".txt")
        hr_txt_src = hr_src_p.with_suffix(".txt")

        lr = read_rgb(lr_src_p)
        hr = read_rgb(hr_src_p)

        lr_proc = resize_with_aspect_and_gray_padding(
            lr,
            out_h=args.lr_h,
            out_w=args.lr_w,
            gray_color=bg,
        )
        hr_proc = resize_with_aspect_and_gray_padding(
            hr,
            out_h=2 * args.lr_h,
            out_w=2 * args.lr_w,
            gray_color=bg,
        )

        write_rgb(lr_dst, lr_proc)
        write_rgb(hr_dst, hr_proc)

        if hr_txt_src.exists():
            hr_txt_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hr_txt_src, hr_txt_dst)
        else:
            # Fallback for raw-train structure: use annotations.json -> plate_text
            ann_src = hr_src_p.parent / "annotations.json"
            if ann_src.exists():
                with open(ann_src, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                plate_text = ann.get("plate_text", "")
                hr_txt_dst.parent.mkdir(parents=True, exist_ok=True)
                with open(hr_txt_dst, "w", encoding="utf-8") as f:
                    f.write(f"plate: {plate_text}\n")
            else:
                missing_txt += 1
                missing_annotations += 1

        out_lines.append(f"{hr_dst.as_posix()};{lr_dst.as_posix()};{split}")

    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[DONE] preprocessed pairs: {len(out_lines)}")
    print(f"[DONE] output root: {out_root}")
    print(f"[DONE] output split: {split_out}")
    print(f"[INFO] missing hr txt labels: {missing_txt}")
    print(f"[INFO] missing annotations fallback: {missing_annotations}")


if __name__ == "__main__":
    main()
