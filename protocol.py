from pathlib import Path

import numpy as np

from .base_dataset import BaseDataset
from .goals_dataset import (
    GOALSDataset,
    ensure_goals_split_csv,
    read_goals_mask,
    read_goals_split_csv,
)
from .nidek_dataset import NidekDataset
from .listing import build_image_mask_lists, list_images


def is_goals_dataset(data_cfg):
    return str(data_cfg.get("dataset", "")).lower() == "goals"


def is_nidek_dataset(data_cfg):
    return str(data_cfg.get("dataset", "")).lower() == "nidek"


def goals_paths(data_cfg):
    root = Path(data_cfg.get("goals_root", "./data/GOALS"))
    image_path = Path(data_cfg.get("train_images", root / "Train/Image"))
    mask_path = Path(data_cfg.get("train_masks", root / "Train/Layer_Masks"))
    return image_path, mask_path


def goals_split_csv(data_cfg):
    split_mode = data_cfg.get("split_mode", "label10")
    seed = int(data_cfg.get("split_seed", 42))
    split_csv = data_cfg.get("split_csv")
    if split_csv:
        return split_csv
    return f"ResSave/splits/goals_{split_mode}_seed{seed}.csv"


def ensure_goals_protocol(data_cfg, force=False):
    image_path, mask_path = goals_paths(data_cfg)
    split_csv = goals_split_csv(data_cfg)
    ensure_goals_split_csv(
        image_path=image_path,
        mask_path=mask_path,
        split_csv=split_csv,
        split_mode=data_cfg.get("split_mode", "label10"),
        seed=int(data_cfg.get("split_seed", 42)),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        labeled_ratio=data_cfg.get("labeled_ratio"),
        force=bool(data_cfg.get("force_split", False) or force),
    )
    return split_csv


def goals_rows(data_cfg, subset):
    split_csv = ensure_goals_protocol(data_cfg)
    return read_goals_split_csv(split_csv, subset=subset)


def goals_filelists(data_cfg, subset):
    rows = goals_rows(data_cfg, subset)
    return [row["filename"] for row in rows]


def goals_image_files(data_cfg, subset):
    rows = goals_rows(data_cfg, subset)
    return [row["image_file"] for row in rows]


def goals_mask_files(data_cfg, subset):
    rows = goals_rows(data_cfg, subset)
    return [row["mask_file"] for row in rows]


def build_supervised_dataset(data_cfg, subset):
    if is_goals_dataset(data_cfg):
        image_path, mask_path = goals_paths(data_cfg)
        return GOALSDataset(
            image_file=str(image_path),
            gt_path=str(mask_path),
            filelists=goals_filelists(data_cfg, subset),
            mode="train",
            resize_to=int(data_cfg.get("image_size", [512, 512])[0]),
            binary_target=data_cfg.get("binary_target", "choroid"),
            augment=subset in ("labeled", "train"),
        )

    if is_nidek_dataset(data_cfg):
        split_file = data_cfg.get("split_file", "ResSave/splits/nidek_patient_seed42.json")
        return NidekDataset(
            split_file=split_file,
            subset=subset,
            labeled_ratio=str(data_cfg.get("labeled_ratio", "0.1")),
            root=data_cfg.get("root"),
            image_size=data_cfg.get("image_size", (512, 512)),
            augment=subset in ("labeled", "train"),
        )

    if subset == "labeled":
        images, masks = build_image_mask_lists(
            data_cfg["train_images"],
            data_cfg["train_masks"],
            exts=data_cfg.get("image_exts"),
            mask_suffix=data_cfg.get("mask_suffix", ""),
        )
    elif subset == "val":
        images, masks = build_image_mask_lists(
            data_cfg["val_images"],
            data_cfg["val_masks"],
            exts=data_cfg.get("image_exts"),
            mask_suffix=data_cfg.get("mask_suffix", ""),
        )
    else:
        raise ValueError(f"Unsupported supervised subset: {subset}")

    return BaseDataset(
        images,
        masks,
        image_size=data_cfg.get("image_size", (512, 512)),
        num_classes=data_cfg.get("num_classes", 2),
    )


def build_unlabeled_dataset(data_cfg):
    if is_goals_dataset(data_cfg):
        image_path, mask_path = goals_paths(data_cfg)
        filelists = goals_filelists(data_cfg, "unlabeled")
        return GOALSDataset(
            image_file=str(image_path),
            gt_path=str(mask_path),
            filelists=filelists,
            mode="test",
            resize_to=int(data_cfg.get("image_size", [512, 512])[0]),
            binary_target=data_cfg.get("binary_target", "choroid"),
        )

    if is_nidek_dataset(data_cfg):
        split_file = data_cfg.get("split_file", "ResSave/splits/nidek_patient_seed42.json")
        return NidekDataset(
            split_file=split_file,
            subset="unlabeled",
            labeled_ratio=str(data_cfg.get("labeled_ratio", "0.1")),
            root=data_cfg.get("root"),
            image_size=data_cfg.get("image_size", (512, 512)),
        )

    image_files = list_images(data_cfg["unlabeled_images"], exts=data_cfg.get("image_exts"))
    return BaseDataset(
        image_files,
        mask_list=None,
        image_size=data_cfg.get("image_size", (512, 512)),
        num_classes=data_cfg.get("num_classes", 2),
    )


def get_labeled_mask_files(data_cfg):
    if is_goals_dataset(data_cfg):
        return goals_mask_files(data_cfg, "labeled")

    if is_nidek_dataset(data_cfg):
        import json
        split_file = data_cfg.get("split_file", "ResSave/splits/nidek_patient_seed42.json")
        labeled_ratio = str(data_cfg.get("labeled_ratio", "0.1"))
        with open(split_file, encoding="utf-8") as f:
            sp = json.load(f)
        root = Path(data_cfg.get("root", sp["meta"]["root"]))
        return [str(root / s["mask"]) for s in sp["labeled"][labeled_ratio]["samples"]]

    _, train_masks = build_image_mask_lists(
        data_cfg["train_images"],
        data_cfg["train_masks"],
        exts=data_cfg.get("image_exts"),
        mask_suffix=data_cfg.get("mask_suffix", ""),
    )
    return train_masks


def load_labeled_binary_masks(data_cfg):
    image_size = data_cfg.get("image_size", (512, 512))
    if is_goals_dataset(data_cfg):
        return [
            read_goals_mask(
                mask_file,
                resize_to=int(image_size[0]),
                binary_target=data_cfg.get("binary_target", "choroid"),
            ).astype(np.float32)
            for mask_file in get_labeled_mask_files(data_cfg)
        ]

    if is_nidek_dataset(data_cfg):
        from PIL import Image
        masks = []
        for mask_file in get_labeled_mask_files(data_cfg):
            arr = np.array(Image.open(mask_file).convert("L"))
            # 自动检测多值标签
            unique = np.unique(arr)
            if len(unique) > 2 and unique.max() > 1 and unique.max() <= 255:
                mask = (arr > 0).astype(np.float32)
            else:
                mask = (arr > 127).astype(np.float32)
            if image_size is not None:
                mask = np.array(
                    Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (image_size[1], image_size[0]), Image.NEAREST)
                ) > 127
                mask = mask.astype(np.float32)
            masks.append(mask)
        return masks

    return None
