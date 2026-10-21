import os
from pathlib import Path
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


image_size = 512

GOALS_ROOT = "./data/GOALS"
GOALS_IMAGE_DIR = "Train/Image"
GOALS_MASK_DIR = "Train/Layer_Masks"

GOALS_LABEL_VALUES = {
    0: 0,    # RNFL
    80: 1,   # GCIPL
    160: 2,  # choroid
    255: 3,  # other
}

GOALS_CLASS_NAMES = {
    "rnfl": 0,
    "gcipl": 1,
    "choroid": 2,
    "other": 3,
    "others": 3,
    "foreground": "foreground",
}


def normalize_split_mode(split_mode):
    mode = str(split_mode).lower().replace("_", "").replace("-", "")
    aliases = {
        "10": "label10",
        "10%": "label10",
        "label10": "label10",
        "labeled10": "label10",
        "labeled10percent": "label10",
        "20": "label20",
        "20%": "label20",
        "label20": "label20",
        "labeled20": "label20",
        "labeled20percent": "label20",
        "full": "full",
        "all": "full",
        "fully": "full",
        "fullysupervised": "full",
    }
    if mode not in aliases:
        raise ValueError("split_mode must be one of: label10, label20, full")
    return aliases[mode]


def labeled_ratio_from_mode(split_mode):
    mode = normalize_split_mode(split_mode)
    if mode == "label10":
        return 0.10
    if mode == "label20":
        return 0.20
    return 1.0


def _normalise_filelists(filelists):
    if filelists is None:
        return None
    return {Path(item).name for item in filelists}


def _list_images(image_path, filelists=None):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_path}")
    allow = _normalise_filelists(filelists)
    files = sorted(
        item.name
        for item in image_path.iterdir()
        if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if allow is not None:
        files = [item for item in files if item in allow]
    return files


def _write_split_csv(split_csv, rows):
    split_csv = Path(split_csv)
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "image_file",
        "mask_file",
        "subset",
        "split_mode",
        "seed",
        "val_ratio",
        "labeled_ratio",
    ]
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_goals_split_csv(split_csv, subset=None):
    rows = []
    with open(split_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if subset is not None and row["subset"] != subset:
                continue
            rows.append(row)
    return rows


def create_goals_split_csv(
    image_path,
    mask_path,
    split_csv,
    split_mode="label10",
    seed=42,
    val_ratio=0.2,
    labeled_ratio=None,
    filelists=None,
):
    """Create the fixed GOALS protocol split.

    The independent validation set is sampled first. The labeled percentage is then
    applied to the remaining training pool.
    """

    split_mode = normalize_split_mode(split_mode)
    labeled_ratio = labeled_ratio_from_mode(split_mode) if labeled_ratio is None else labeled_ratio
    names = _list_images(image_path, filelists=filelists)
    if not names:
        raise ValueError(f"No images found in {image_path}")

    rng = np.random.default_rng(int(seed))
    order = np.arange(len(names))
    rng.shuffle(order)

    val_count = int(round(len(names) * float(val_ratio)))
    val_count = min(max(1, val_count), len(names) - 1)
    val_ids = set(order[:val_count].tolist())
    train_ids = order[val_count:].tolist()

    if split_mode == "full":
        labeled_count = len(train_ids)
    else:
        labeled_count = int(round(len(train_ids) * float(labeled_ratio)))
        labeled_count = min(max(1, labeled_count), len(train_ids))

    labeled_ids = set(train_ids[:labeled_count])
    unlabeled_ids = set(train_ids[labeled_count:])
    mask_path = Path(mask_path)
    image_path = Path(image_path)

    rows = []
    for idx, name in enumerate(names):
        if idx in val_ids:
            subset = "val"
        elif idx in labeled_ids:
            subset = "labeled"
        elif idx in unlabeled_ids:
            subset = "unlabeled"
        else:
            raise RuntimeError(f"Image index {idx} was not assigned to a split")

        rows.append(
            {
                "filename": name,
                "image_file": str(image_path / name),
                "mask_file": str(mask_path / name),
                "subset": subset,
                "split_mode": split_mode,
                "seed": int(seed),
                "val_ratio": float(val_ratio),
                "labeled_ratio": float(labeled_ratio),
            }
        )

    _write_split_csv(split_csv, rows)
    return rows


def ensure_goals_split_csv(
    image_path,
    mask_path,
    split_csv,
    split_mode="label10",
    seed=42,
    val_ratio=0.2,
    labeled_ratio=None,
    force=False,
    filelists=None,
):
    split_csv = Path(split_csv)
    if force or not split_csv.exists():
        create_goals_split_csv(
            image_path=image_path,
            mask_path=mask_path,
            split_csv=split_csv,
            split_mode=split_mode,
            seed=seed,
            val_ratio=val_ratio,
            labeled_ratio=labeled_ratio,
            filelists=filelists,
        )
    return split_csv


def convert_goals_mask(mask_img):
    """Map official GOALS layer values 0/80/160/255 to class ids 0/1/2/3."""

    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 1]
    mapped = np.zeros(mask_img.shape, dtype=np.uint8)
    for raw_value, class_id in GOALS_LABEL_VALUES.items():
        mapped[mask_img == raw_value] = class_id
    return mapped


def binary_from_goals_mask(mask, binary_target=None):
    if binary_target is None:
        return mask
    if isinstance(binary_target, str):
        binary_target = GOALS_CLASS_NAMES[binary_target.lower()]
    if binary_target == "foreground":
        return (mask > 0).astype(np.uint8)
    return (mask == int(binary_target)).astype(np.uint8)


def read_goals_image(img_path, resize_to=image_size):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w, c = img.shape
    if resize_to is not None:
        img = cv2.resize(img, (int(resize_to), int(resize_to)), interpolation=cv2.INTER_LINEAR)
    return img, h, w, c


def read_goals_mask(mask_path, resize_to=image_size, binary_target=None):
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    mask = convert_goals_mask(mask_img)
    if resize_to is not None:
        mask = cv2.resize(mask, (int(resize_to), int(resize_to)), interpolation=cv2.INTER_NEAREST)
    mask = binary_from_goals_mask(mask, binary_target=binary_target)
    return mask.astype(np.int64)


class OCTDataset(Dataset):
    """Official-style GOALS OCT dataset.

    In train mode returns:
        img, gt_img

    In test mode returns:
        img, real_index, original_h, original_w

    Images are returned as C,H,W numpy arrays, following the official interface.
    """

    def __init__(
        self,
        image_file,
        gt_path=None,
        filelists=None,
        mode="train",
        resize_to=image_size,
        binary_target=None,
    ):
        super().__init__()
        self.mode = mode
        self.image_path = image_file
        self.gt_path = gt_path
        self.resize_to = resize_to
        self.binary_target = binary_target
        self.file_list = _list_images(self.image_path, filelists=filelists)

        if self.mode == "train" and self.gt_path is None:
            raise ValueError("gt_path is required when mode='train'")

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        img, h, w, _ = read_goals_image(img_path, resize_to=self.resize_to)
        img = img.transpose(2, 0, 1)

        if self.mode == "test":
            return img, real_index, h, w

        gt_tmp_path = os.path.join(self.gt_path, real_index)
        gt_img = read_goals_mask(
            gt_tmp_path,
            resize_to=self.resize_to,
            binary_target=self.binary_target,
        )
        return img, gt_img

    def __len__(self):
        return len(self.file_list)


class GOALSDataset(Dataset):
    """Prior-SSL sample-dict wrapper for the GOALS dataset."""

    def __init__(
        self,
        image_file=None,
        gt_path=None,
        filelists=None,
        mode="train",
        subset=None,
        split_mode="label10",
        split_csv=None,
        seed=42,
        val_ratio=0.2,
        labeled_ratio=None,
        force_split=False,
        resize_to=image_size,
        binary_target=None,
        normalize=True,
        augment=False,
        flip_prob=0.5,
        contrast_range=(0.9, 1.1),
    ):
        super().__init__()
        root = Path(GOALS_ROOT)
        self.image_path = str(Path(image_file) if image_file else root / GOALS_IMAGE_DIR)
        self.gt_path = str(Path(gt_path) if gt_path else root / GOALS_MASK_DIR)
        self.mode = mode
        self.resize_to = resize_to
        self.binary_target = binary_target
        self.normalize = normalize
        self.augment = augment
        self.flip_prob = float(flip_prob)
        self.contrast_range = tuple(contrast_range)
        self.subset = subset
        self.split_mode = normalize_split_mode(split_mode)

        if subset is not None:
            split_csv = split_csv or self._default_split_csv(self.split_mode, seed)
            ensure_goals_split_csv(
                self.image_path,
                self.gt_path,
                split_csv,
                split_mode=self.split_mode,
                seed=seed,
                val_ratio=val_ratio,
                labeled_ratio=labeled_ratio,
                force=force_split,
                filelists=filelists,
            )
            rows = read_goals_split_csv(split_csv, subset=subset)
            self.file_list = [row["filename"] for row in rows]
        else:
            self.file_list = _list_images(self.image_path, filelists=filelists)

    @staticmethod
    def _default_split_csv(split_mode, seed):
        return f"ResSave/splits/goals_{normalize_split_mode(split_mode)}_seed{seed}.csv"

    def _augment_pair(self, img, mask=None):
        if torch.rand(1).item() < self.flip_prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        low, high = self.contrast_range
        if high > low:
            factor = low + (high - low) * torch.rand(1).item()
            img_f = img.astype(np.float32)
            mean = float(img_f.mean())
            img = np.clip((img_f - mean) * factor + mean, 0, 255).astype(np.uint8)

        return img, None if mask is None else np.ascontiguousarray(mask)

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        img_path = os.path.join(self.image_path, real_index)
        img, h, w, _ = read_goals_image(img_path, resize_to=self.resize_to)

        # cv2 loads BGR; convert to RGB before augmentation and normalization.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        mask_path = None
        if self.mode != "test":
            mask_path = os.path.join(self.gt_path, real_index)
            mask = read_goals_mask(
                mask_path,
                resize_to=self.resize_to,
                binary_target=self.binary_target,
            )

        if self.augment:
            img, mask = self._augment_pair(img, mask)

        img = img.astype(np.float32) / 255.0
        if self.normalize:
            img = (img - 0.5) / 0.5

        image = torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ).float()

        sample = {
            "image": image,
            "image_file": str(img_path),
            "file_name": real_index,
            "height": h,
            "width": w,
        }

        if self.mode == "test":
            return sample

        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        sample.update(
            {
                "mask": mask,
                "mask_file": str(mask_path),
                "is_pseudo": torch.tensor(False),
                "confidence": torch.ones_like(mask, dtype=torch.float32),
                "confidence_file": "",
            }
        )
        return sample

    def __len__(self):
        return len(self.file_list)


class GOALS_dataset:
    """Small helper matching the older dataset-helper style used in experiments."""

    def __init__(
        self,
        root=GOALS_ROOT,
        image_dir=GOALS_IMAGE_DIR,
        mask_dir=GOALS_MASK_DIR,
    ):
        self.root = Path(root)
        self.image_path = self.root / image_dir
        self.mask_path = self.root / mask_dir

    def getFileNames(self):
        return _list_images(self.image_path)

    def getImageList(self, filelists=None):
        names = _list_images(self.image_path, filelists=filelists)
        return [str(self.image_path / name) for name in names]

    def getMaskList(self, image_files):
        mask_files = []
        for image_file in image_files:
            name = Path(image_file).name
            mask_file = self.mask_path / name
            if not mask_file.exists():
                raise FileNotFoundError(f"No GOALS mask found for {name}: {mask_file}")
            mask_files.append(str(mask_file))
        return mask_files

    def create_split_csv(
        self,
        split_csv,
        split_mode="label10",
        seed=42,
        val_ratio=0.2,
        labeled_ratio=None,
        force=True,
    ):
        if force or not Path(split_csv).exists():
            return create_goals_split_csv(
                self.image_path,
                self.mask_path,
                split_csv,
                split_mode=split_mode,
                seed=seed,
                val_ratio=val_ratio,
                labeled_ratio=labeled_ratio,
            )
        return read_goals_split_csv(split_csv)

    def ensure_split_csv(
        self,
        split_csv,
        split_mode="label10",
        seed=42,
        val_ratio=0.2,
        labeled_ratio=None,
        force=False,
    ):
        return ensure_goals_split_csv(
            self.image_path,
            self.mask_path,
            split_csv,
            split_mode=split_mode,
            seed=seed,
            val_ratio=val_ratio,
            labeled_ratio=labeled_ratio,
            force=force,
        )

    def get_split_rows(self, split_csv, subset=None):
        return read_goals_split_csv(split_csv, subset=subset)

    def get_split_lists(self, split_csv, subset):
        rows = self.get_split_rows(split_csv, subset=subset)
        image_files = [row["image_file"] for row in rows]
        mask_files = [row["mask_file"] for row in rows]
        return image_files, mask_files

    def split_train_val(self, train_ratio=0.8, seed=42):
        names = self.getFileNames()
        rng = np.random.default_rng(seed)
        order = np.arange(len(names))
        rng.shuffle(order)
        split_idx = int(len(names) * float(train_ratio))
        train_names = [names[i] for i in order[:split_idx]]
        val_names = [names[i] for i in order[split_idx:]]
        return {
            "train": self.getImageList(train_names),
            "val": self.getImageList(val_names),
        }

    def KFold_train_vali(self, k=5, seed=42):
        names = self.getFileNames()
        rng = np.random.default_rng(seed)
        order = np.arange(len(names))
        rng.shuffle(order)
        folds = np.array_split(order, int(k))

        split = {}
        for fold_idx in range(int(k)):
            val_ids = set(folds[fold_idx].tolist())
            train_names = [name for idx, name in enumerate(names) if idx not in val_ids]
            val_names = [name for idx, name in enumerate(names) if idx in val_ids]
            split[str(fold_idx)] = [
                self.getImageList(train_names),
                self.getImageList(val_names),
            ]
        return split


if __name__ == "__main__":
    dataset = GOALS_dataset()
    files = dataset.getFileNames()
    print(f"GOALS files: {len(files)}")
    if files:
        oct_dataset = OCTDataset(
            image_file=str(dataset.image_path),
            gt_path=str(dataset.mask_path),
            filelists=[files[0]],
            mode="train",
        )
        img, mask = oct_dataset[0]
        print("official-style image:", img.shape, "mask:", mask.shape, np.unique(mask))

        ssl_dataset = GOALSDataset(filelists=[files[0]], mode="train", binary_target="choroid")
        sample = ssl_dataset[0]
        print("Prior-SSL sample:", sample["image"].shape, sample["mask"].shape)
