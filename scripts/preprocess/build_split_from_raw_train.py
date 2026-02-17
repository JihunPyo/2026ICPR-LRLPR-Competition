#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

# Ensure project root is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Build split txt from raw-train track structure.")
    p.add_argument("--train-root", required=True, help="Path to extracted train directory (contains Scenario-*/.../track_*)")
    p.add_argument("--out-split", required=True, help="Output split txt path")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio by track")
    p.add_argument("--seed", type=int, default=1996)
    p.add_argument("--frame-index", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Use one lr/hr frame index per track")
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="Use all 5 lr/hr frames per track (keeps track-level train/val split).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    train_root = Path(args.train_root)
    tracks = []
    for scenario_dir in sorted(train_root.glob("Scenario-*")):
        for layout_dir in sorted(scenario_dir.glob("*")):
            if not layout_dir.is_dir():
                continue
            for track_dir in sorted(layout_dir.glob("track_*")):
                ann = track_dir / "annotations.json"
                if not ann.exists():
                    continue

                frame_indices = [1, 2, 3, 4, 5] if args.all_frames else [args.frame_index]
                pairs = []
                for idx in frame_indices:
                    lr = track_dir / f"lr-{idx:03d}.jpg"
                    hr = track_dir / f"hr-{idx:03d}.jpg"
                    if lr.exists() and hr.exists():
                        pairs.append((hr.as_posix(), lr.as_posix()))

                if pairs:
                    tracks.append(pairs)

    if not tracks:
        raise RuntimeError(f"No valid tracks found under: {train_root}")

    rng = random.Random(args.seed)
    rng.shuffle(tracks)
    split_idx = int(len(tracks) * (1.0 - args.val_ratio))

    train_tracks = tracks[:split_idx]
    val_tracks = tracks[split_idx:]

    out_lines = []
    for pairs in train_tracks:
        out_lines.extend(f"{hr};{lr};training" for hr, lr in pairs)
    for pairs in val_tracks:
        out_lines.extend(f"{hr};{lr};validation" for hr, lr in pairs)

    out_path = Path(args.out_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    train_items = sum(len(x) for x in train_tracks)
    val_items = sum(len(x) for x in val_tracks)
    print(
        f"[DONE] tracks total={len(tracks)} train={len(train_tracks)} val={len(val_tracks)} | "
        f"samples total={train_items + val_items} train={train_items} val={val_items}"
    )
    print(f"[DONE] split file: {out_path}")
    if args.all_frames:
        print("[INFO] frame indices used: 1..5")
    else:
        print(f"[INFO] frame index used: {args.frame_index}")


if __name__ == "__main__":
    main()
