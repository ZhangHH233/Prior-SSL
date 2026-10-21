import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .datasets.goals_dataset import read_goals_mask


def binary_mask_from_file(mask_file, image_size=None, goals=False, binary_target="choroid"):
    if goals:
        resize_to = int(image_size[0]) if image_size is not None else None
        return read_goals_mask(
            mask_file,
            resize_to=resize_to,
            binary_target=binary_target,
        ).astype(np.uint8)

    mask = Image.open(mask_file).convert("L")
    if image_size is not None:
        mask = mask.resize(tuple(image_size), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def pseudo_mask_from_file(mask_file, image_size=None):
    mask = Image.open(mask_file).convert("L")
    if image_size is not None:
        mask = mask.resize(tuple(image_size), Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8)


def gt_mask_for_image(image_file, data_cfg):
    mask_root = Path(data_cfg["train_masks"])
    return mask_root / Path(image_file).name


def dice_coefficient(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    cardinality = pred.sum() + target.sum()
    return float((2.0 * intersection + eps) / (cardinality + eps))


def thickness_profile(mask):
    return mask.astype(np.float32).sum(axis=0)


def thickness_summary(mask):
    profile = thickness_profile(mask)
    return {
        "cht_mean": float(profile.mean()),
        "cht_std": float(profile.std()),
        "cht_max": float(profile.max()),
        "cht_area": float(profile.sum()),
    }


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(x_rank, y_rank)


def regression_line(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept
    return xs, ys


def mae(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[valid] - y[valid])))


def rmse(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[valid] - y[valid]) ** 2)))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
