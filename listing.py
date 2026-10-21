from pathlib import Path
import csv


def list_images(root, exts=None):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    exts = exts or ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pattern in exts:
        files.extend(root.glob(pattern))
    return sorted(str(p) for p in files)


def build_image_mask_lists(image_root, mask_root, exts=None, mask_suffix=""):
    image_files = list_images(image_root, exts=exts)
    mask_root = Path(mask_root)
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")

    mask_files = []
    for image_file in image_files:
        image_path = Path(image_file)
        candidates = [
            mask_root / f"{image_path.stem}{mask_suffix}{image_path.suffix}",
            mask_root / f"{image_path.stem}{mask_suffix}.png",
            mask_root / f"{image_path.stem}.png",
            mask_root / image_path.name,
        ]
        mask_file = next((p for p in candidates if p.exists()), None)
        if mask_file is None:
            raise FileNotFoundError(
                f"No mask found for {image_path.name}. Tried: "
                + ", ".join(str(p) for p in candidates)
            )
        mask_files.append(str(mask_file))

    return image_files, mask_files


def _read_confidence_score(row):
    for key in (
        "selection_score",
        "final_fg_mean",
        "final_nonzero_mean",
        "mean_confidence",
    ):
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return 0.0


def read_pseudo_index(index_csv, min_confidence=0.0):
    image_files = []
    pseudo_mask_files = []
    confidence_files = []

    with open(index_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence_score = _read_confidence_score(row)
            keep_value = row.get("keep", "1")
            keep = keep_value in {"1", "true", "True", "yes", "YES"}
            if not keep or confidence_score < min_confidence:
                continue
            image_files.append(row["image_file"])
            pseudo_mask_files.append(row["pseudo_mask_file"])
            confidence_files.append(row["confidence_file"])

    return image_files, pseudo_mask_files, confidence_files
