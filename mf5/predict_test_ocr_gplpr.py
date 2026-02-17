import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models


def normalize_state_dict_keys(sd: dict):
    if not sd:
        return sd
    has_module_prefix = all(k.startswith("module.") for k in sd.keys())
    if not has_module_prefix:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def clean_plate_text(text: str) -> str:
    return str(text).replace("#", "").replace("-", "").replace(" ", "").strip()


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


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


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


class SRTrackDataset(Dataset):
    def __init__(self, sr_root: str):
        root = Path(sr_root)
        samples = []
        for track_dir in sorted(root.glob("track_*")):
            sr_path = track_dir / "sr.png"
            if sr_path.exists():
                samples.append({"track_id": track_dir.name, "sr_path": sr_path})
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["sr_path"]).convert("RGB")
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(32, 96), mode="bicubic", align_corners=False).squeeze(0)
        return {"track_id": item["track_id"], "sr": x}

    @staticmethod
    def collate_fn(batch):
        track_id = [b["track_id"] for b in batch]
        sr = torch.stack([b["sr"] for b in batch], dim=0)
        return {"track_id": track_id, "sr": sr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr-dir", required=True, help="Directory from mf5 inference (contains track_x/sr.png).")
    parser.add_argument("--ocr-checkpoint", required=True, help="GPLPR checkpoint path.")
    parser.add_argument("--output-txt", required=True, help="Output txt path.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ocr-alphabet", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--ocr-nc", type=int, default=3)
    parser.add_argument("--ocr-k", type=int, default=7)
    parser.add_argument("--ocr-is-seq-model", type=str2bool, default=True)
    parser.add_argument("--ocr-head", type=int, default=2)
    parser.add_argument("--ocr-inner", type=int, default=256)
    parser.add_argument("--ocr-is-l2-norm", type=str2bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args = load_gplpr_model(
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

    dataset = SRTrackDataset(args.sr_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=SRTrackDataset.collate_fn,
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ocr-test", leave=False):
            sr = batch["sr"].to(device, non_blocking=True)
            _, logits, _ = model(sr)
            pred_idx = logits.argmax(2).detach().cpu()
            pred_texts = [clean_plate_text(t) for t in converter.decode_list(pred_idx)]
            probs = torch.softmax(logits, dim=2).detach().cpu()
            max_probs = probs.max(dim=2).values
            confs = max_probs.mean(dim=1)
            for track_id, pred, conf in zip(batch["track_id"], pred_texts, confs):
                rows.append({"track_id": track_id, "pred_text": pred, "confidence": float(conf)})

    rows = sorted(rows, key=lambda x: x["track_id"])
    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['track_id']},{row['pred_text']};{row['confidence']:.4f}\n")

    print(f"processed tracks: {len(rows)}")
    print(f"saved txt: {out_path}")


if __name__ == "__main__":
    main()
