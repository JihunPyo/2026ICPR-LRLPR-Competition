import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from mf5.data import TrackSequenceDataset, TrackSequenceWrapper


def tensor_to_pil(img_tensor):
    img = img_tensor.detach().cpu().clamp(0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_set = TrackSequenceDataset(data_root=config["data"]["root"], phase="testing")
    test_set = TrackSequenceWrapper(
        dataset=base_set,
        imgW=config["preprocess"]["imgW"],
        imgH=config["preprocess"]["imgH"],
        aug=False,
        image_aspect_ratio=config["preprocess"]["image_aspect_ratio"],
        background=config["preprocess"]["background"],
        phase="testing",
        input_mode=config["preprocess"].get("input_mode", "stack15"),
    )

    loader = DataLoader(
        test_set,
        batch_size=config["infer"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["infer"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=TrackSequenceWrapper.collate_fn,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_spec = checkpoint["model"]
    model = models.make({"name": model_spec["name"], "args": model_spec["args"]})
    model.load_state_dict(model_spec["sd"], strict=True)
    model = model.to(device)
    model.eval()

    save_dir = Path(config["infer"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", leave=False):
            lr = batch["lr"].to(device, non_blocking=True)
            pred = model(lr)

            for track_id, sr_img in zip(batch["track_id"], pred):
                track_dir = save_dir / track_id
                track_dir.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(sr_img).save(track_dir / "sr.png")

    print(f"saved results to: {save_dir}")


if __name__ == "__main__":
    main()
