import ast
from typing import Tuple

import cv2
import numpy as np


def parse_background_color(background) -> Tuple[int, int, int]:
    if isinstance(background, str):
        parsed = ast.literal_eval(background)
    else:
        parsed = background

    if isinstance(parsed, (int, float)):
        v = int(round(float(parsed)))
        return (v, v, v)

    if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
        vals = []
        for v in parsed:
            fv = float(v)
            if 0.0 <= fv <= 1.0:
                fv *= 255.0
            vals.append(int(round(fv)))
        return tuple(vals)

    raise ValueError(f"Invalid background color: {background}")


def resize_with_aspect_and_gray_padding(
    img: np.ndarray,
    out_h: int,
    out_w: int,
    gray_color: Tuple[int, int, int] = (127, 127, 127),
) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={img.shape}")

    in_h, in_w = img.shape[:2]
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid input image size: {img.shape}")

    scale = min(out_w / in_w, out_h / in_h)
    new_w = max(1, int(round(in_w * scale)))
    new_h = max(1, int(round(in_h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((out_h, out_w, 3), gray_color, dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
