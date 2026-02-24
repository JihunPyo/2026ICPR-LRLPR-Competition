import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss


def clean_plate_text(text: str) -> str:
    return str(text).replace("#", "").replace("-", "").strip()


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


class LossPackMF5(nn.Module):
    def __init__(
        self,
        alphabet: str,
        k: int = 7,
        ce_weight: float = 1.0,
        ssim_weight: float = 0.75,
        layout_weight: float = 0.2,
        ssim_window: int = 5,
        confusing_pair_weight: float = 0.5,
        ce_impl: str = "onehot",
    ):
        super().__init__()
        self.converter = StrLabelConverter(alphabet)
        self.k = int(k)
        self.ce_weight = float(ce_weight)
        self.ssim_weight = float(ssim_weight)
        self.layout_weight = float(layout_weight)
        self.confusing_pair_weight = float(confusing_pair_weight)
        self.ce_impl = str(ce_impl).lower()
        if self.ce_impl not in {"onehot", "logits"}:
            raise ValueError(f"Unsupported ce_impl: {ce_impl}. Supported: onehot, logits")
        self.ssim = SSIMLoss(window_size=int(ssim_window)) if self.ssim_weight > 0 else None
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
            a = str(a)
            b = str(b)
            if a == b:
                continue
            if (a not in self.converter.dict) or (b not in self.converter.dict):
                continue
            normalized.add(tuple(sorted((a, b))))
        self.confusing_pairs = normalized

    def _is_visually_confusing(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return False
        return tuple(sorted((a, b))) in self.confusing_pairs

    def _build_sample_weights(self, pred_texts, gt_texts, device, dtype):
        num_classes = len(self.converter.alphabet)
        weights = torch.ones((len(gt_texts), num_classes), device=device, dtype=dtype)
        if not self.confusing_pairs:
            return weights
        for i, (pred_text, gt_text) in enumerate(zip(pred_texts, gt_texts)):
            pred_text = clean_plate_text(pred_text)
            gt_text = clean_plate_text(gt_text)
            for pred_char, gt_char in zip(pred_text, gt_text):
                if self._is_visually_confusing(pred_char, gt_char):
                    idx = self.converter.dict.get(gt_char)
                    if idx is None or idx == 0:
                        continue
                    weights[i, int(idx)] = weights[i, int(idx)] + self.confusing_pair_weight
        return weights

    def _ce_logits(self, logits: torch.Tensor, target: torch.Tensor, sample_weights: torch.Tensor):
        if logits.size(0) == 0:
            return logits.new_zeros(())
        per_sample = []
        for i in range(logits.size(0)):
            per_sample.append(
                F.cross_entropy(
                    logits[i],
                    target[i],
                    weight=sample_weights[i],
                )
            )
        return torch.stack(per_sample).mean()

    def _ce_onehot(self, logits: torch.Tensor, target: torch.Tensor, sample_weights: torch.Tensor):
        if logits.size(0) == 0:
            return logits.new_zeros(())
        pred_indices = logits.argmax(2)
        pred_onehot = F.one_hot(pred_indices, num_classes=logits.size(-1)).to(dtype=logits.dtype)
        per_sample = []
        for i in range(pred_onehot.size(0)):
            per_sample.append(
                F.cross_entropy(
                    pred_onehot[i],
                    target[i],
                    weight=sample_weights[i],
                )
            )
        return torch.stack(per_sample).mean()

    def _compute_ssim_term(self, pred_sr: torch.Tensor, hr_stack: torch.Tensor):
        if self.ssim is None:
            return pred_sr.new_zeros(())
        ssim_losses = [self.ssim(pred_sr, hr_stack[:, i]) for i in range(hr_stack.size(1))]
        return torch.stack(ssim_losses).mean()

    def forward(self, pred_sr: torch.Tensor, hr_stack: torch.Tensor, ocr_logits: torch.Tensor, gt_texts):
        gt_texts = [clean_plate_text(t) for t in gt_texts]
        target = self.converter.encode_list(gt_texts, k=self.k).to(ocr_logits.device)
        pred_indices = ocr_logits.argmax(2).detach().cpu()
        pred_texts = [clean_plate_text(t) for t in self.converter.decode_list(pred_indices)]
        sample_weights = self._build_sample_weights(
            pred_texts=pred_texts,
            gt_texts=gt_texts,
            device=ocr_logits.device,
            dtype=ocr_logits.dtype,
        )

        if self.ce_impl == "logits":
            ce_loss = self._ce_logits(ocr_logits, target, sample_weights)
        else:
            ce_loss = self._ce_onehot(ocr_logits, target, sample_weights)

        penalty = 0.0
        for pred, gt in zip(pred_texts, gt_texts):
            penalty += self.layout_penalty(pred, gt)
        penalty = penalty / max(1, len(gt_texts))
        layout_penalty = ocr_logits.new_tensor(penalty)

        ssim_loss = self._compute_ssim_term(pred_sr, hr_stack)
        total = (
            self.ce_weight * ce_loss
            + self.ssim_weight * ssim_loss
            + self.layout_weight * layout_penalty
        )
        return total, ce_loss.detach(), ssim_loss.detach(), layout_penalty.detach()
