import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from kornia.losses import SSIMLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import models
from mf5.data import TrackSequenceDataset, TrackSequenceWrapper

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    return distributed, local_rank, world_size


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(distributed: bool) -> bool:
    return (not distributed) or dist.get_rank() == 0


def clean_plate_text(text: str) -> str:
    return str(text).replace("#", "").replace("-", "").strip()


def normalize_state_dict_keys(sd: dict):
    if not sd:
        return sd
    has_module_prefix = all(k.startswith("module.") for k in sd.keys())
    if not has_module_prefix:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


class StrLabelConverter:
    def __init__(self, alphabet: str):
        self.alphabet = "-" + alphabet
        self.dict = {char: i for i, char in enumerate(self.alphabet)}

    def encode_char(self, char: str) -> int:
        return self.dict[char]

    def encode_list(self, texts, k: int = 7):
        encoded = []
        for text in texts:
            text = clean_plate_text(text)
            row = []
            for i in range(k):
                if i < len(text) and text[i] in self.dict:
                    row.append(self.dict[text[i]])
                else:
                    row.append(0)
            encoded.append(row)
        return torch.LongTensor(encoded)

    def decode_list(self, indices: torch.Tensor):
        outputs = []
        for row in indices:
            chars = []
            for idx in row:
                idx_int = int(idx)
                if idx_int == 0:
                    continue
                chars.append(self.alphabet[idx_int])
            outputs.append("".join(chars))
        return outputs


class LCOFLLoss(nn.Module):
    def __init__(
        self,
        alphabet: str,
        k: int = 7,
        layout_alpha: float = 1.0,
        confusing_pair_weight: float = 0.5,
    ):
        super().__init__()
        self.converter = StrLabelConverter(alphabet)
        self.k = k
        self.layout_alpha = layout_alpha
        self.confusing_pair_weight = float(confusing_pair_weight)
        self.confusing_pairs = set()

    @staticmethod
    def layout_penalty(pred_layout: str, gt_layout: str):
        penalty = 0.0
        for pred_char, gt_char in zip(pred_layout, gt_layout):
            if pred_char.isdigit() and gt_char.isalpha():
                penalty += 0.4
            elif pred_char.isalpha() and gt_char.isdigit():
                penalty += 0.5
        return penalty

    def set_confusing_pairs(self, pairs):
        normalized = set()
        for a, b in pairs:
            if not a or not b:
                continue
            normalized.add((str(a), str(b)))
        self.confusing_pairs = normalized

    def weighted_ce(self, logits: torch.Tensor, target: torch.Tensor, pred_texts, gt_texts):
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).reshape(target.shape)
        if self.confusing_pairs:
            weights = torch.ones_like(ce)
            for i, (pred, gt) in enumerate(zip(pred_texts, gt_texts)):
                for j, (pred_char, gt_char) in enumerate(zip(pred, gt)):
                    if (pred_char, gt_char) in self.confusing_pairs or (
                        gt_char,
                        pred_char,
                    ) in self.confusing_pairs:
                        weights[i, j] = weights[i, j] + self.confusing_pair_weight
            ce = ce * weights
        return ce.mean()

    def forward(self, logits: torch.Tensor, gt_texts):
        target = self.converter.encode_list(gt_texts, k=self.k).to(logits.device)

        pred_idx = logits.argmax(2).detach().cpu()
        pred_texts = [clean_plate_text(t) for t in self.converter.decode_list(pred_idx)]
        gt_texts = [clean_plate_text(t) for t in gt_texts]
        ce_loss = self.weighted_ce(logits, target, pred_texts, gt_texts)

        penalty = 0.0
        for pred, gt in zip(pred_texts, gt_texts):
            penalty += self.layout_penalty(pred, gt)
        penalty = penalty / max(1, len(gt_texts))
        penalty = logits.new_tensor(penalty)

        total = ce_loss + self.layout_alpha * penalty
        return total, ce_loss.detach(), penalty.detach()


def extract_confusing_pairs(conf_matrix, alphabet: str, threshold: float = 0.25):
    pairs = []
    class_names = "-" + alphabet
    for i in range(1, len(class_names)):  # skip blank
        for j in range(1, len(class_names)):
            if i != j and conf_matrix[i, j] > threshold:
                pairs.append((class_names[i], class_names[j]))
    return pairs


def build_confusion_matrix(pred_indices, gt_indices, num_classes: int):
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    if not pred_indices:
        return conf
    pred_all = np.concatenate(pred_indices).reshape(-1)
    gt_all = np.concatenate(gt_indices).reshape(-1)
    valid = (pred_all >= 0) & (pred_all < num_classes) & (gt_all >= 0) & (gt_all < num_classes)
    pred_all = pred_all[valid]
    gt_all = gt_all[valid]
    np.add.at(conf, (pred_all, gt_all), 1)
    return conf


class MultiHRLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.0, ssim_window: int = 5):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(window_size=ssim_window) if ssim_weight > 0 else None

    def forward(self, pred, target):
        loss = self.l1_weight * self.l1(pred, target)
        if self.ssim is not None:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        return loss


def make_model(config, device):
    model_cfg = config["model"]
    model = models.make({"name": model_cfg["name"], "args": model_cfg["args"]})

    ckpt_path = model_cfg.get("checkpoint")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and "sd" in ckpt["model"]:
            src_sd = ckpt["model"]["sd"]
        elif isinstance(ckpt, dict):
            src_sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        tgt_sd = model.state_dict()
        load_sd = {}
        for k, v in src_sd.items():
            if k not in tgt_sd:
                continue
            if v.shape == tgt_sd[k].shape:
                load_sd[k] = v
                continue
            # Typical finetuning case: pretrained 3ch -> target 15ch at first conv.
            if (
                v.ndim == 4
                and tgt_sd[k].ndim == 4
                and v.shape[0] == tgt_sd[k].shape[0]
                and v.shape[2:] == tgt_sd[k].shape[2:]
                and v.shape[1] == 3
                and tgt_sd[k].shape[1] % 3 == 0
            ):
                rep = tgt_sd[k].shape[1] // 3
                load_sd[k] = v.repeat(1, rep, 1, 1) / rep

        missing, unexpected = model.load_state_dict(load_sd, strict=False)
        print(
            f"Loaded pretrained checkpoint: {ckpt_path} | "
            f"matched={len(load_sd)} missing={len(missing)} unexpected={len(unexpected)}"
        )

    return model.to(device)


def load_gplpr_model(config, device):
    ocr_cfg = config.get("ocr", {})
    checkpoint_path = ocr_cfg.get("ocr_ckpt")
    if checkpoint_path is None:
        raise ValueError("`ocr.ocr_ckpt` must be set when `loss.lcofl_weight > 0`")

    model_args = {
        "alphabet": ocr_cfg.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "nc": int(ocr_cfg.get("nc", 3)),
        "K": int(ocr_cfg.get("k", 7)),
        "isSeqModel": bool(ocr_cfg.get("is_seq_model", True)),
        "head": int(ocr_cfg.get("head", 2)),
        "inner": int(ocr_cfg.get("inner", 256)),
        "isl2Norm": bool(ocr_cfg.get("is_l2_norm", True)),
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

    return model, model_args, checkpoint_path


def reduce_mean(value: torch.Tensor, distributed: bool):
    if not distributed:
        return value
    with torch.no_grad():
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def set_requires_grad(model, flag: bool):
    if model is None:
        return
    for param in unwrap_model(model).parameters():
        param.requires_grad_(flag)


def select_real_hr_frame(hr: torch.Tensor, mode: str):
    # hr: [B, T, C, H, W]
    if mode == "center":
        return hr[:, hr.size(1) // 2]
    if mode == "avg":
        return hr.mean(dim=1)
    if mode == "random":
        b = hr.size(0)
        idx = torch.randint(0, hr.size(1), (b,), device=hr.device)
        return hr[torch.arange(b, device=hr.device), idx]
    raise ValueError(f"Unsupported adv.real_frame_mode: {mode}")


def prepare_ocr_input(x: torch.Tensor):
    if x.shape[-2:] != (32, 96):
        x = F.interpolate(x, size=(32, 96), mode="bicubic", align_corners=False)
    return x


def compute_ocr_ce_loss(logits: torch.Tensor, gt_texts, converter: StrLabelConverter, k: int):
    target = converter.encode_list(gt_texts, k=k).to(logits.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def compute_lcofl_loss(pred, gt_texts, ocr_model, lcofl_criterion, lcofl_weight):
    zero = pred.new_zeros(())
    if lcofl_weight <= 0.0 or ocr_model is None or lcofl_criterion is None:
        return zero, zero, zero

    ocr_in = prepare_ocr_input(pred)

    _, logits, _ = ocr_model(ocr_in)
    return lcofl_criterion(logits, gt_texts)


def get_lcofl_weight_for_epoch(base_weight: float, epoch: int, schedule_cfg):
    if not schedule_cfg:
        return float(base_weight)
    if not bool(schedule_cfg.get("enabled", False)):
        return float(base_weight)

    start_weight = float(schedule_cfg.get("start_weight", base_weight))
    end_weight = float(schedule_cfg.get("end_weight", base_weight))
    start_epoch = int(schedule_cfg.get("start_epoch", 1))
    end_epoch = int(schedule_cfg.get("end_epoch", start_epoch))
    mode = str(schedule_cfg.get("mode", "linear")).lower()

    if end_epoch <= start_epoch:
        return float(end_weight if epoch >= start_epoch else start_weight)
    if epoch <= start_epoch:
        return float(start_weight)
    if epoch >= end_epoch:
        return float(end_weight)

    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    if mode == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif mode != "linear":
        raise ValueError(f"Unsupported lcofl schedule mode: {mode}")
    return float(start_weight + (end_weight - start_weight) * progress)


def train_one_epoch(
    model,
    ocr_model,
    loader,
    optimizer,
    ocr_optimizer,
    scaler,
    criterion,
    lcofl_criterion,
    ocr_converter,
    ocr_k,
    device,
    update_mode,
    pixel_weight,
    lcofl_weight,
    adv_enabled,
    adv_weight,
    adv_d_steps,
    adv_real_frame_mode,
    adv_start_epoch,
    epoch,
    amp_enabled,
    distributed,
):
    model.train()
    if ocr_model is not None and not (adv_enabled and ocr_optimizer is not None):
        ocr_model.eval()

    total_loss = 0.0
    total_pixel_loss = 0.0
    total_lcofl_loss = 0.0
    total_lcofl_ce = 0.0
    total_layout_penalty = 0.0
    total_adv_g_loss = 0.0
    total_d_loss = 0.0
    total_d_real_ce = 0.0
    total_d_fake_ce = 0.0
    n_batches = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = tqdm(loader, disable=not is_main_process(distributed), desc="train", leave=False)

    for batch in pbar:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        gt_texts = batch.get("gt", [])
        adv_active = (
            adv_enabled
            and adv_weight > 0.0
            and ocr_model is not None
            and ocr_optimizer is not None
            and ocr_converter is not None
            and epoch >= adv_start_epoch
        )

        if ocr_model is not None and not adv_active:
            set_requires_grad(ocr_model, False)
            unwrap_model(ocr_model).eval()

        if adv_active:
            set_requires_grad(ocr_model, True)
            unwrap_model(ocr_model).train()
            d_loss_steps = []
            d_real_steps = []
            d_fake_steps = []
            for _ in range(max(1, int(adv_d_steps))):
                with torch.no_grad():
                    pred_for_d = model(lr)
                real_for_d = select_real_hr_frame(hr, adv_real_frame_mode)
                fake_for_d = pred_for_d.detach()
                real_for_d = prepare_ocr_input(real_for_d)
                fake_for_d = prepare_ocr_input(fake_for_d)

                ocr_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_type, enabled=amp_enabled):
                    _, logits_real_d, _ = ocr_model(real_for_d)
                    _, logits_fake_d, _ = ocr_model(fake_for_d)
                    d_real_ce = compute_ocr_ce_loss(logits_real_d, gt_texts, ocr_converter, ocr_k)
                    d_fake_ce = compute_ocr_ce_loss(logits_fake_d, gt_texts, ocr_converter, ocr_k)
                    d_loss = d_real_ce + d_fake_ce
                scaler.scale(d_loss).backward()
                scaler.step(ocr_optimizer)
                scaler.update()

                d_loss_steps.append(d_loss.detach())
                d_real_steps.append(d_real_ce.detach())
                d_fake_steps.append(d_fake_ce.detach())

            d_loss_tensor = torch.stack(d_loss_steps).mean()
            d_real_tensor = torch.stack(d_real_steps).mean()
            d_fake_tensor = torch.stack(d_fake_steps).mean()
            set_requires_grad(ocr_model, False)
            unwrap_model(ocr_model).eval()
        else:
            zero = lr.new_zeros(())
            d_loss_tensor = zero
            d_real_tensor = zero
            d_fake_tensor = zero

        if update_mode == "per_hr_step":
            step_total_losses = []
            step_pixel_losses = []
            step_lcofl_losses = []
            step_lcofl_ce_losses = []
            step_layout_penalties = []
            step_adv_g_losses = []
            for i in range(hr.size(1)):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_type, enabled=amp_enabled):
                    pred = model(lr)
                    pixel_loss = criterion(pred, hr[:, i])
                    lcofl_loss, lcofl_ce, layout_penalty = compute_lcofl_loss(
                        pred, gt_texts, ocr_model, lcofl_criterion, lcofl_weight
                    )
                    if (
                        adv_enabled
                        and adv_weight > 0.0
                        and ocr_model is not None
                        and ocr_converter is not None
                        and epoch >= adv_start_epoch
                    ):
                        _, logits_fake_g, _ = ocr_model(prepare_ocr_input(pred))
                        adv_g_loss = compute_ocr_ce_loss(logits_fake_g, gt_texts, ocr_converter, ocr_k)
                    else:
                        adv_g_loss = pred.new_zeros(())
                    loss = (
                        pixel_weight * pixel_loss
                        + lcofl_weight * lcofl_loss
                        + adv_weight * adv_g_loss
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_total_losses.append(loss.detach())
                step_pixel_losses.append(pixel_loss.detach())
                step_lcofl_losses.append(lcofl_loss.detach())
                step_lcofl_ce_losses.append(lcofl_ce.detach())
                step_layout_penalties.append(layout_penalty.detach())
                step_adv_g_losses.append(adv_g_loss.detach())
            loss_tensor = torch.stack(step_total_losses).mean()
            pixel_loss_tensor = torch.stack(step_pixel_losses).mean()
            lcofl_loss_tensor = torch.stack(step_lcofl_losses).mean()
            lcofl_ce_tensor = torch.stack(step_lcofl_ce_losses).mean()
            layout_penalty_tensor = torch.stack(step_layout_penalties).mean()
            adv_g_tensor = torch.stack(step_adv_g_losses).mean()
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                pred = model(lr)
                pixel_loss = torch.stack([criterion(pred, hr[:, i]) for i in range(hr.size(1))]).mean()
                lcofl_loss, lcofl_ce, layout_penalty = compute_lcofl_loss(
                    pred, gt_texts, ocr_model, lcofl_criterion, lcofl_weight
                )
                if (
                    adv_enabled
                    and adv_weight > 0.0
                    and ocr_model is not None
                    and ocr_converter is not None
                    and epoch >= adv_start_epoch
                ):
                    _, logits_fake_g, _ = ocr_model(prepare_ocr_input(pred))
                    adv_g_loss = compute_ocr_ce_loss(logits_fake_g, gt_texts, ocr_converter, ocr_k)
                else:
                    adv_g_loss = pred.new_zeros(())
                loss = (
                    pixel_weight * pixel_loss
                    + lcofl_weight * lcofl_loss
                    + adv_weight * adv_g_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_tensor = loss.detach()
            pixel_loss_tensor = pixel_loss.detach()
            lcofl_loss_tensor = lcofl_loss.detach()
            lcofl_ce_tensor = lcofl_ce.detach()
            layout_penalty_tensor = layout_penalty.detach()
            adv_g_tensor = adv_g_loss.detach()

        loss_tensor = reduce_mean(loss_tensor, distributed)
        pixel_loss_tensor = reduce_mean(pixel_loss_tensor, distributed)
        lcofl_loss_tensor = reduce_mean(lcofl_loss_tensor, distributed)
        lcofl_ce_tensor = reduce_mean(lcofl_ce_tensor, distributed)
        layout_penalty_tensor = reduce_mean(layout_penalty_tensor, distributed)
        adv_g_tensor = reduce_mean(adv_g_tensor, distributed)
        d_loss_tensor = reduce_mean(d_loss_tensor, distributed)
        d_real_tensor = reduce_mean(d_real_tensor, distributed)
        d_fake_tensor = reduce_mean(d_fake_tensor, distributed)

        total_loss += loss_tensor.item()
        total_pixel_loss += pixel_loss_tensor.item()
        total_lcofl_loss += lcofl_loss_tensor.item()
        total_lcofl_ce += lcofl_ce_tensor.item()
        total_layout_penalty += layout_penalty_tensor.item()
        total_adv_g_loss += adv_g_tensor.item()
        total_d_loss += d_loss_tensor.item()
        total_d_real_ce += d_real_tensor.item()
        total_d_fake_ce += d_fake_tensor.item()
        n_batches += 1
        pbar.set_postfix(
            {
                "loss": total_loss / max(1, n_batches),
                "pix": total_pixel_loss / max(1, n_batches),
                "lcofl": total_lcofl_loss / max(1, n_batches),
                "adv_g": total_adv_g_loss / max(1, n_batches),
                "d": total_d_loss / max(1, n_batches),
            }
        )

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "pixel_loss": total_pixel_loss / denom,
        "lcofl_loss": total_lcofl_loss / denom,
        "lcofl_ce": total_lcofl_ce / denom,
        "layout_penalty": total_layout_penalty / denom,
        "adv_g_loss": total_adv_g_loss / denom,
        "d_loss": total_d_loss / denom,
        "d_real_ce": total_d_real_ce / denom,
        "d_fake_ce": total_d_fake_ce / denom,
    }


def validate_one_epoch(
    model,
    ocr_model,
    loader,
    criterion,
    lcofl_criterion,
    ocr_converter,
    ocr_k,
    device,
    pixel_weight,
    lcofl_weight,
    adv_enabled,
    adv_weight,
    adv_start_epoch,
    epoch,
    amp_enabled,
    distributed,
    cm_enabled=False,
    cm_threshold=0.25,
    cm_alphabet="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
):
    model.eval()
    if ocr_model is not None:
        ocr_model.eval()

    total_loss = 0.0
    total_pixel_loss = 0.0
    total_lcofl_loss = 0.0
    total_lcofl_ce = 0.0
    total_layout_penalty = 0.0
    total_adv_g_loss = 0.0
    n_batches = 0
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    preds_cm = []
    gts_cm = []

    pbar = tqdm(loader, disable=not is_main_process(distributed), desc="val", leave=False)

    with torch.no_grad():
        for batch in pbar:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            gt_texts = batch.get("gt", [])

            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                pred = model(lr)
                pixel_loss = torch.stack([criterion(pred, hr[:, i]) for i in range(hr.size(1))]).mean()
                lcofl_loss, lcofl_ce, layout_penalty = compute_lcofl_loss(
                    pred, gt_texts, ocr_model, lcofl_criterion, lcofl_weight
                )
                if cm_enabled and ocr_model is not None:
                    _, logits_fake_cm, _ = ocr_model(prepare_ocr_input(pred))
                    pred_idx_cm = logits_fake_cm.argmax(2).detach().cpu().numpy()
                    gt_idx_cm = ocr_converter.encode_list(gt_texts, k=ocr_k).cpu().numpy()
                    preds_cm.append(pred_idx_cm)
                    gts_cm.append(gt_idx_cm)
                if (
                    adv_enabled
                    and adv_weight > 0.0
                    and ocr_model is not None
                    and ocr_converter is not None
                    and epoch >= adv_start_epoch
                ):
                    _, logits_fake_g, _ = ocr_model(prepare_ocr_input(pred))
                    adv_g_loss = compute_ocr_ce_loss(logits_fake_g, gt_texts, ocr_converter, ocr_k)
                else:
                    adv_g_loss = pred.new_zeros(())
                loss = (
                    pixel_weight * pixel_loss
                    + lcofl_weight * lcofl_loss
                    + adv_weight * adv_g_loss
                )

            loss = reduce_mean(loss.detach(), distributed)
            pixel_loss = reduce_mean(pixel_loss.detach(), distributed)
            lcofl_loss = reduce_mean(lcofl_loss.detach(), distributed)
            lcofl_ce = reduce_mean(lcofl_ce.detach(), distributed)
            layout_penalty = reduce_mean(layout_penalty.detach(), distributed)
            adv_g_loss = reduce_mean(adv_g_loss.detach(), distributed)
            total_loss += loss.item()
            total_pixel_loss += pixel_loss.item()
            total_lcofl_loss += lcofl_loss.item()
            total_lcofl_ce += lcofl_ce.item()
            total_layout_penalty += layout_penalty.item()
            total_adv_g_loss += adv_g_loss.item()
            n_batches += 1
            pbar.set_postfix(
                {
                    "loss": total_loss / max(1, n_batches),
                    "pix": total_pixel_loss / max(1, n_batches),
                    "lcofl": total_lcofl_loss / max(1, n_batches),
                    "adv_g": total_adv_g_loss / max(1, n_batches),
                }
            )

    denom = max(1, n_batches)
    confusing_pairs = []
    if cm_enabled and (not distributed):
        num_classes = len("-" + cm_alphabet)
        conf = build_confusion_matrix(preds_cm, gts_cm, num_classes=num_classes)
        conf_norm = conf.astype(np.float64) / (conf.sum(axis=1, keepdims=True) + 1e-10)
        confusing_pairs = extract_confusing_pairs(conf_norm, cm_alphabet, threshold=cm_threshold)
    return {
        "loss": total_loss / denom,
        "pixel_loss": total_pixel_loss / denom,
        "lcofl_loss": total_lcofl_loss / denom,
        "lcofl_ce": total_lcofl_ce / denom,
        "layout_penalty": total_layout_penalty / denom,
        "adv_g_loss": total_adv_g_loss / denom,
        "d_loss": 0.0,
        "d_real_ce": 0.0,
        "d_fake_ce": 0.0,
        "confusing_pairs": confusing_pairs,
    }


def save_checkpoint(
    save_dir: Path,
    epoch: int,
    model,
    optimizer,
    best_val: float,
    config: dict,
    name: str,
    ocr_model=None,
    ocr_model_args=None,
    ocr_optimizer=None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": {
            "name": config["model"]["name"],
            "args": config["model"]["args"],
            "sd": unwrap_model(model).state_dict(),
        },
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "config": config,
    }
    if ocr_model is not None:
        payload["ocr_model"] = {
            "name": "GPLPR",
            "args": ocr_model_args or {},
            "sd": unwrap_model(ocr_model).state_dict(),
        }
    if ocr_optimizer is not None:
        payload["ocr_optimizer"] = ocr_optimizer.state_dict()
    torch.save(payload, save_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    distributed, local_rank, _ = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    setup_seed(config.get("seed", 1996))
    wandb_cfg = config.get("wandb", {})

    train_base = TrackSequenceDataset(
        data_root=config["data"]["root"],
        phase="training",
        val_ratio=config["data"].get("val_ratio", 0.1),
        seed=config.get("seed", 1996),
        scenario_filter=config["data"].get("scenario_filter"),
        layout_filter=config["data"].get("layout_filter"),
    )
    val_base = TrackSequenceDataset(
        data_root=config["data"]["root"],
        phase="validation",
        val_ratio=config["data"].get("val_ratio", 0.1),
        seed=config.get("seed", 1996),
        scenario_filter=config["data"].get("scenario_filter"),
        layout_filter=config["data"].get("layout_filter"),
    )

    train_set = TrackSequenceWrapper(
        dataset=train_base,
        imgW=config["preprocess"]["imgW"],
        imgH=config["preprocess"]["imgH"],
        aug=config["preprocess"].get("aug", True),
        image_aspect_ratio=config["preprocess"]["image_aspect_ratio"],
        background=config["preprocess"]["background"],
        phase="training",
        input_mode=config["preprocess"].get("input_mode", "stack15"),
    )
    val_set = TrackSequenceWrapper(
        dataset=val_base,
        imgW=config["preprocess"]["imgW"],
        imgH=config["preprocess"]["imgH"],
        aug=False,
        image_aspect_ratio=config["preprocess"]["image_aspect_ratio"],
        background=config["preprocess"]["background"],
        phase="validation",
        input_mode=config["preprocess"].get("input_mode", "stack15"),
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    loader_args = {
        "num_workers": config["train"].get("num_workers", 4),
        "pin_memory": True,
        "collate_fn": TrackSequenceWrapper.collate_fn,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_args,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["train"]["val_batch_size"],
        shuffle=False,
        sampler=val_sampler,
        **loader_args,
    )

    model = make_model(config, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = MultiHRLoss(
        l1_weight=config["loss"].get("l1_weight", 1.0),
        ssim_weight=config["loss"].get("ssim_weight", 0.0),
        ssim_window=config["loss"].get("ssim_window", 5),
    ).to(device)
    pixel_weight = float(config["loss"].get("pixel_weight", 1.0))
    lcofl_weight = float(config["loss"].get("lcofl_weight", 0.0))
    lcofl_schedule_cfg = config["loss"].get("lcofl_schedule", {})
    cm_cfg = config["loss"].get("confusion_matrix", {})
    cm_enabled = bool(cm_cfg.get("enabled", False))
    cm_threshold = float(cm_cfg.get("threshold", 0.25))
    cm_pair_weight = float(cm_cfg.get("pair_weight", 0.5))
    adv_cfg = config.get("adv", {})
    adv_enabled = bool(adv_cfg.get("enabled", False))
    adv_weight = float(adv_cfg.get("g_weight", adv_cfg.get("weight", 0.0)))
    adv_d_steps = int(adv_cfg.get("d_steps", 1))
    adv_real_frame_mode = adv_cfg.get("real_frame_mode", "random")
    adv_start_epoch = int(adv_cfg.get("warmup_epochs", 0)) + 1

    ocr_model = None
    lcofl_criterion = None
    ocr_converter = None
    ocr_k = 7
    ocr_optimizer = None
    ocr_scheduler = None
    ocr_model_args = None
    ocr_checkpoint = None
    need_ocr = (lcofl_weight > 0.0) or adv_enabled
    if need_ocr:
        ocr_train = bool(config.get("ocr", {}).get("ocr_train", False))
        if adv_enabled and not ocr_train:
            raise ValueError("Adversarial mode requires `ocr.ocr_train=true` to update discriminator.")
        if (not adv_enabled) and ocr_train:
            raise ValueError("`ocr_train=true` without adversarial mode is not supported.")

        ocr_model, ocr_model_args, ocr_checkpoint = load_gplpr_model(config, device)
        ocr_k = int(ocr_model_args.get("K", 7))
        ocr_converter = StrLabelConverter(
            ocr_model_args.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        )

        lcofl_criterion = LCOFLLoss(
            alphabet=ocr_model_args.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            k=ocr_k,
            layout_alpha=float(config.get("ocr", {}).get("layout_alpha", 1.0)),
            confusing_pair_weight=cm_pair_weight,
        ).to(device)

        if distributed:
            ocr_model = DDP(ocr_model, device_ids=[local_rank], output_device=local_rank)

        if ocr_train:
            ocr_optimizer = torch.optim.Adam(
                ocr_model.parameters(),
                lr=float(adv_cfg.get("d_lr", config["train"]["lr"])),
                betas=tuple(adv_cfg.get("d_betas", [0.9, 0.999])),
                weight_decay=float(adv_cfg.get("d_weight_decay", 0.0)),
            )
            ocr_scheduler = torch.optim.lr_scheduler.StepLR(
                ocr_optimizer,
                step_size=int(adv_cfg.get("d_lr_step", config["train"].get("lr_step", 10))),
                gamma=float(adv_cfg.get("d_lr_gamma", config["train"].get("lr_gamma", 0.5))),
            )
        else:
            set_requires_grad(ocr_model, False)
            unwrap_model(ocr_model).eval()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        betas=tuple(config["train"].get("betas", [0.9, 0.999])),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["train"].get("lr_step", 10),
        gamma=config["train"].get("lr_gamma", 0.5),
    )

    amp_enabled = bool(config["train"].get("amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val = float("inf")
    resume_path = config["train"].get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        target_model = model.module if distributed else model
        target_model.load_state_dict(checkpoint["model"]["sd"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ocr_model is not None and "ocr_model" in checkpoint:
            unwrap_model(ocr_model).load_state_dict(checkpoint["ocr_model"]["sd"], strict=True)
        if ocr_optimizer is not None and "ocr_optimizer" in checkpoint:
            ocr_optimizer.load_state_dict(checkpoint["ocr_optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)

    save_dir = Path(config["train"]["save_dir"])
    metrics_path = save_dir / "metrics.csv"
    metrics_jsonl_path = save_dir / "metrics.jsonl"
    save_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process(distributed) and not metrics_path.exists():
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_pixel_loss",
                    "train_lcofl_loss",
                    "train_lcofl_ce",
                    "train_layout_penalty",
                    "train_adv_g_loss",
                    "train_d_loss",
                    "train_d_real_ce",
                    "train_d_fake_ce",
                    "lcofl_weight",
                    "val_loss",
                    "val_pixel_loss",
                    "val_lcofl_loss",
                    "val_lcofl_ce",
                    "val_layout_penalty",
                    "val_adv_g_loss",
                    "gap",
                    "best_val",
                ]
            )
    if is_main_process(distributed) and not metrics_jsonl_path.exists():
        metrics_jsonl_path.touch()

    if is_main_process(distributed):
        print(f"train samples: {len(train_set)}")
        print(f"val samples: {len(val_set)}")
        print(f"update mode: {config['train'].get('update_mode', 'avg5_once')}")
        print(f"pixel weight: {pixel_weight}")
        print(f"lcofl weight: {lcofl_weight}")
        if lcofl_schedule_cfg and bool(lcofl_schedule_cfg.get("enabled", False)):
            print(f"lcofl schedule: {lcofl_schedule_cfg}")
        print(f"confusion matrix update: {cm_enabled}")
        if cm_enabled:
            print(f"cm threshold: {cm_threshold}")
            print(f"cm pair weight: {cm_pair_weight}")
            if distributed:
                print("[WARN] cm update is disabled in distributed mode.")
        print(f"adversarial enabled: {adv_enabled}")
        print(f"adversarial weight: {adv_weight}")
        if adv_enabled:
            print(f"adversarial d_steps: {adv_d_steps}")
            print(f"adversarial warmup epochs: {adv_start_epoch - 1}")
            print(f"adversarial real frame mode: {adv_real_frame_mode}")
        if need_ocr:
            print(f"ocr checkpoint: {ocr_checkpoint}")
            print(f"ocr args: {ocr_model_args}")
            print(f"ocr trainable: {ocr_optimizer is not None}")
        print(f"metrics csv: {metrics_path}")
        print(f"metrics jsonl: {metrics_jsonl_path}")

    use_wandb = bool(wandb_cfg.get("enabled", False))
    if use_wandb and wandb is None and is_main_process(distributed):
        print("[WARN] wandb is enabled in config but package is not installed.")
        use_wandb = False

    if use_wandb and is_main_process(distributed):
        run_name = wandb_cfg.get("run_name") or f"mf5-{Path(config['train']['save_dir']).name}"
        try:
            wandb.init(
                project=wandb_cfg.get("project", "lpr-mf5"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config=config,
                dir=wandb_cfg.get("dir"),
                mode=wandb_cfg.get("mode", "online"),
            )
        except Exception as e:
            print(f"[WARN] wandb init failed, continue without wandb: {e}")
            use_wandb = False

    epochs = config["train"]["epochs"]
    worsen_count = 0
    for epoch in range(start_epoch, epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
        lcofl_weight_epoch = get_lcofl_weight_for_epoch(lcofl_weight, epoch, lcofl_schedule_cfg)

        train_metrics = train_one_epoch(
            model=model,
            ocr_model=ocr_model,
            loader=train_loader,
            optimizer=optimizer,
            ocr_optimizer=ocr_optimizer,
            scaler=scaler,
            criterion=criterion,
            lcofl_criterion=lcofl_criterion,
            ocr_converter=ocr_converter,
            ocr_k=ocr_k,
            device=device,
            update_mode=config["train"].get("update_mode", "avg5_once"),
            pixel_weight=pixel_weight,
            lcofl_weight=lcofl_weight_epoch,
            adv_enabled=adv_enabled,
            adv_weight=adv_weight,
            adv_d_steps=adv_d_steps,
            adv_real_frame_mode=adv_real_frame_mode,
            adv_start_epoch=adv_start_epoch,
            epoch=epoch,
            amp_enabled=amp_enabled,
            distributed=distributed,
        )
        val_metrics = validate_one_epoch(
            model=model,
            ocr_model=ocr_model,
            loader=val_loader,
            criterion=criterion,
            lcofl_criterion=lcofl_criterion,
            ocr_converter=ocr_converter,
            ocr_k=ocr_k,
            device=device,
            pixel_weight=pixel_weight,
            lcofl_weight=lcofl_weight_epoch,
            adv_enabled=adv_enabled,
            adv_weight=adv_weight,
            adv_start_epoch=adv_start_epoch,
            epoch=epoch,
            amp_enabled=amp_enabled,
            distributed=distributed,
            cm_enabled=(cm_enabled and (not distributed)),
            cm_threshold=cm_threshold,
            cm_alphabet=ocr_model_args.get("alphabet", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            if ocr_model_args is not None
            else "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
        if lcofl_criterion is not None and cm_enabled and (not distributed):
            lcofl_criterion.set_confusing_pairs(val_metrics.get("confusing_pairs", []))
        scheduler.step()
        if ocr_scheduler is not None:
            ocr_scheduler.step()

        if is_main_process(distributed):
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            lr_now = optimizer.param_groups[0]["lr"]
            gap = val_loss - train_loss
            if val_loss > best_val:
                worsen_count += 1
            else:
                worsen_count = 0
            print(
                f"epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_pix={train_metrics['pixel_loss']:.6f} val_pix={val_metrics['pixel_loss']:.6f} "
                f"train_lcofl={train_metrics['lcofl_loss']:.6f} val_lcofl={val_metrics['lcofl_loss']:.6f} "
                f"train_adv_g={train_metrics['adv_g_loss']:.6f} val_adv_g={val_metrics['adv_g_loss']:.6f} "
                f"train_d={train_metrics['d_loss']:.6f} "
                f"lcofl_w={lcofl_weight_epoch:.4f} "
                f"cm_pairs={len(val_metrics.get('confusing_pairs', []))} "
                f"gap={gap:+.6f} lr={lr_now:.2e}"
            )
            if worsen_count >= 3 and gap > 0:
                print(
                    f"[WARN] possible overfitting: val loss worsened for {worsen_count} epochs "
                    f"and gap is positive ({gap:+.6f})"
                )

            model_to_save = model.module if distributed else model
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    save_dir,
                    epoch,
                    model_to_save,
                    optimizer,
                    best_val,
                    config,
                    "best.pth",
                    ocr_model=ocr_model,
                    ocr_model_args=ocr_model_args,
                    ocr_optimizer=ocr_optimizer,
                )
            save_checkpoint(
                save_dir,
                epoch,
                model_to_save,
                optimizer,
                best_val,
                config,
                "last.pth",
                ocr_model=ocr_model,
                ocr_model_args=ocr_model_args,
                ocr_optimizer=ocr_optimizer,
            )

            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        lr_now,
                        train_loss,
                        train_metrics["pixel_loss"],
                        train_metrics["lcofl_loss"],
                        train_metrics["lcofl_ce"],
                        train_metrics["layout_penalty"],
                        train_metrics["adv_g_loss"],
                        train_metrics["d_loss"],
                        train_metrics["d_real_ce"],
                        train_metrics["d_fake_ce"],
                        lcofl_weight_epoch,
                        val_loss,
                        val_metrics["pixel_loss"],
                        val_metrics["lcofl_loss"],
                        val_metrics["lcofl_ce"],
                        val_metrics["layout_penalty"],
                        val_metrics["adv_g_loss"],
                        gap,
                        best_val,
                    ]
                )
            with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "lr": lr_now,
                            "train_loss": train_loss,
                            "train_pixel_loss": train_metrics["pixel_loss"],
                            "train_lcofl_loss": train_metrics["lcofl_loss"],
                            "train_lcofl_ce": train_metrics["lcofl_ce"],
                            "train_layout_penalty": train_metrics["layout_penalty"],
                            "train_adv_g_loss": train_metrics["adv_g_loss"],
                            "train_d_loss": train_metrics["d_loss"],
                            "train_d_real_ce": train_metrics["d_real_ce"],
                            "train_d_fake_ce": train_metrics["d_fake_ce"],
                            "lcofl_weight": lcofl_weight_epoch,
                            "val_loss": val_loss,
                            "val_pixel_loss": val_metrics["pixel_loss"],
                            "val_lcofl_loss": val_metrics["lcofl_loss"],
                            "val_lcofl_ce": val_metrics["lcofl_ce"],
                            "val_layout_penalty": val_metrics["layout_penalty"],
                            "val_adv_g_loss": val_metrics["adv_g_loss"],
                            "gap": gap,
                            "best_val": best_val,
                        }
                    )
                    + "\n"
                )

            if use_wandb:
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "lr": lr_now,
                            "train_loss": train_loss,
                            "train_pixel_loss": train_metrics["pixel_loss"],
                            "train_lcofl_loss": train_metrics["lcofl_loss"],
                            "train_lcofl_ce": train_metrics["lcofl_ce"],
                            "train_layout_penalty": train_metrics["layout_penalty"],
                            "train_adv_g_loss": train_metrics["adv_g_loss"],
                            "train_d_loss": train_metrics["d_loss"],
                            "train_d_real_ce": train_metrics["d_real_ce"],
                            "train_d_fake_ce": train_metrics["d_fake_ce"],
                            "lcofl_weight": lcofl_weight_epoch,
                            "val_loss": val_loss,
                            "val_pixel_loss": val_metrics["pixel_loss"],
                            "val_lcofl_loss": val_metrics["lcofl_loss"],
                            "val_lcofl_ce": val_metrics["lcofl_ce"],
                            "val_layout_penalty": val_metrics["layout_penalty"],
                            "val_adv_g_loss": val_metrics["adv_g_loss"],
                            "gap": gap,
                            "best_val": best_val,
                        },
                        step=epoch,
                    )
                except Exception as e:
                    print(f"[WARN] wandb log failed, disable wandb for remaining epochs: {e}")
                    use_wandb = False

    cleanup_distributed(distributed)
    if use_wandb and is_main_process(distributed):
        wandb.finish()


if __name__ == "__main__":
    main()
