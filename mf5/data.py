import json
import random
from pathlib import Path
from typing import Dict, List, Optional

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional runtime dependency
    A = None
import cv2
import kornia as K
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.preprocess import parse_background_color, resize_with_aspect_and_gray_padding

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )


class TrackSequenceDataset(Dataset):
    """Track-level dataset for 5-frame LR/HR pairs."""

    def __init__(
        self,
        data_root: str,
        phase: str,
        val_ratio: float = 0.1,
        seed: int = 1996,
        scenario_filter: Optional[List[str]] = None,
        layout_filter: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.phase = phase
        self.val_ratio = val_ratio
        self.seed = seed
        self.scenario_filter = set(scenario_filter or [])
        self.layout_filter = set(layout_filter or [])
        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        if self.phase in {"training", "validation"}:
            all_samples = self._collect_train_tracks()
            rng = random.Random(self.seed)
            rng.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1.0 - self.val_ratio))
            if self.phase == "training":
                return all_samples[:split_idx]
            return all_samples[split_idx:]

        if self.phase == "testing":
            return self._collect_test_tracks()

        raise ValueError(f"Unknown phase: {self.phase}")

    @staticmethod
    def _find_frame(track_dir: Path, prefix: str, frame_idx: int):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = track_dir / f"{prefix}-{frame_idx:03d}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _collect_train_tracks(self) -> List[Dict]:
        tracks = []
        train_root = self.data_root / "train"

        for scenario_dir in sorted(train_root.glob("Scenario-*")):
            scenario_name = scenario_dir.name
            if self.scenario_filter and scenario_name not in self.scenario_filter:
                continue

            for layout_dir in sorted(scenario_dir.glob("*")):
                if not layout_dir.is_dir():
                    continue
                layout_name = layout_dir.name
                if self.layout_filter and layout_name not in self.layout_filter:
                    continue

                for track_dir in sorted(layout_dir.glob("track_*")):
                    lr_paths = []
                    hr_paths = []
                    valid_frames = True
                    for i in range(1, 6):
                        lr_p = self._find_frame(track_dir, "lr", i)
                        hr_p = self._find_frame(track_dir, "hr", i)
                        if lr_p is None or hr_p is None:
                            valid_frames = False
                            break
                        lr_paths.append(lr_p)
                        hr_paths.append(hr_p)

                    anno_path = track_dir / "annotations.json"
                    if not anno_path.exists():
                        continue
                    if not valid_frames:
                        continue

                    with open(anno_path, "r", encoding="utf-8") as f:
                        anno = json.load(f)

                    tracks.append(
                        {
                            "track_id": track_dir.name,
                            "scenario": scenario_name,
                            "layout": anno.get("plate_layout", layout_name),
                            "gt": anno.get("plate_text", ""),
                            "lr_paths": [str(p) for p in lr_paths],
                            "hr_paths": [str(p) for p in hr_paths],
                        }
                    )

        return tracks

    def _collect_test_tracks(self) -> List[Dict]:
        test_root = self.data_root / "test-public"
        tracks = []

        for track_dir in sorted(test_root.glob("track_*")):
            lr_paths = []
            valid_frames = True
            for i in range(1, 6):
                lr_p = self._find_frame(track_dir, "lr", i)
                if lr_p is None:
                    valid_frames = False
                    break
                lr_paths.append(lr_p)

            if not valid_frames:
                continue
            tracks.append(
                {
                    "track_id": track_dir.name,
                    "gt": "",
                    "lr_paths": [str(p) for p in lr_paths],
                    "hr_paths": [],
                }
            )

        return tracks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class TrackSequenceWrapper(Dataset):
    """Apply original project preprocessing, then stack 5 LR frames into 15 channels."""

    def __init__(
        self,
        dataset: Dataset,
        imgW: int,
        imgH: int,
        aug: bool,
        image_aspect_ratio: float,
        background: str,
        phase: str,
        input_mode: str = "stack15",
    ):
        self.dataset = dataset
        self.imgW = imgW
        self.imgH = imgH
        self.aug = aug
        self.ar = image_aspect_ratio
        self.background = parse_background_color(background)
        self.phase = phase
        self.input_mode = input_mode

        # Keep the same augmentation family as the original wrapper.
        self.transforms = [
            None,
        ]

        if A is not None:
            self.transforms = [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    brightness_by_max=True,
                    always_apply=True,
                    p=1.0,
                ),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0),
                A.InvertImg(always_apply=True),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    always_apply=True,
                    p=1.0,
                ),
                A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1.0),
                None,
            ]

    def __len__(self) -> int:
        return len(self.dataset)

    def _open_image(self, image_path: str) -> "cv2.Mat":
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _augment_lr_frames(self, frames: List) -> List:
        if (not self.aug) or (A is None):
            return frames

        augment = random.choice(self.transforms)
        if augment is None:
            return frames

        replay_compose = A.ReplayCompose([augment])
        first = replay_compose(image=frames[0])
        replay = first["replay"]
        out = [first["image"]]

        for frame in frames[1:]:
            out.append(A.ReplayCompose.replay(replay, image=frame)["image"])

        return out

    def _process_lr(self, img):
        img = resize_with_aspect_and_gray_padding(
            img,
            out_h=self.imgH,
            out_w=self.imgW,
            gray_color=self.background,
        )
        return resize_fn(img, (self.imgH, self.imgW))

    def _process_hr(self, img):
        hr = K.enhance.equalize_clahe(
            transforms.ToTensor()(Image.fromarray(img)).unsqueeze(0),
            clip_limit=4.0,
            grid_size=(2, 2),
        )
        hr = K.utils.tensor_to_image(hr.mul(255.0).byte())
        hr = resize_with_aspect_and_gray_padding(
            hr,
            out_h=2 * self.imgH,
            out_w=2 * self.imgW,
            gray_color=self.background,
        )
        return resize_fn(hr, (2 * self.imgH, 2 * self.imgW))

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]

        lr_frames = [self._open_image(p) for p in item["lr_paths"]]
        lr_frames = self._augment_lr_frames(lr_frames)
        lr_tensors = [self._process_lr(frame) for frame in lr_frames]
        if self.input_mode == "center3":
            lr_input = lr_tensors[2]
        else:
            lr_input = torch.cat(lr_tensors, dim=0)

        out = {
            "lr": lr_input,
            "gt": item.get("gt", ""),
            "track_id": item["track_id"],
        }

        if self.phase != "testing":
            hr_frames = [self._open_image(p) for p in item["hr_paths"]]
            hr_tensors = [self._process_hr(frame) for frame in hr_frames]
            out["hr"] = torch.stack(hr_tensors, dim=0)

        return out

    @staticmethod
    def collate_fn(datas: List[Dict]) -> Dict:
        lr = torch.stack([d["lr"] for d in datas], dim=0)
        track_id = [d["track_id"] for d in datas]
        gt = [d.get("gt", "") for d in datas]

        batch = {
            "lr": lr,
            "track_id": track_id,
            "gt": gt,
        }

        if "hr" in datas[0]:
            batch["hr"] = torch.stack([d["hr"] for d in datas], dim=0)

        return batch
