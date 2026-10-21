import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from prior_ssl.datasets.protocol import (
    build_unlabeled_dataset,
    get_labeled_mask_files,
    load_labeled_binary_masks,
)
from prior_ssl.models import build_model
from prior_ssl.prior import ThicknessMemoryBank
from prior_ssl.utils import (
    ensure_dir,
    load_checkpoint,
    load_config,
    resolve_device,
    set_seed,
    write_csv,
)
from prior_ssl.visualization import probability_from_logits


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo masks and confidence maps")
    parser.add_argument("--config", default="configs/prior_ssl.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def save_mask(mask_array, path, num_classes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if num_classes <= 2:
        mask_array = (mask_array > 0).astype(np.uint8) * 255
    else:
        mask_array = mask_array.astype(np.uint8)
    Image.fromarray(mask_array).save(path)


def save_confidence_png(confidence, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confidence = np.clip(confidence, 0.0, 1.0)
    Image.fromarray((confidence * 255).astype(np.uint8)).save(path)


def apply_perturbation(images, mode):
    if mode == "hflip":
        return torch.flip(images, dims=[3])
    if mode in {"none", None}:
        return images
    raise ValueError(f"Unsupported perturbation: {mode}")


def invert_perturbation(probabilities, mode):
    if mode == "hflip":
        return torch.flip(probabilities, dims=[2])
    if mode in {"none", None}:
        return probabilities
    raise ValueError(f"Unsupported perturbation: {mode}")


def uncertainty_confidence(
    model,
    images,
    num_classes,
    perturbation,
    uncertainty_upper_bound,
    beta,
    eps=1e-6,
):
    logits = model(images)
    prob = probability_from_logits(logits, num_classes=num_classes)

    perturbed_images = apply_perturbation(images, perturbation)
    perturbed_logits = model(perturbed_images)
    perturbed_prob = probability_from_logits(perturbed_logits, num_classes=num_classes)
    aligned_perturbed_prob = invert_perturbation(perturbed_prob, perturbation)

    p = prob.clamp(eps, 1.0 - eps)
    q = aligned_perturbed_prob.clamp(eps, 1.0 - eps)
    d_uc = -p * torch.log(q) - (1.0 - p) * torch.log(1.0 - q)
    upper = max(float(uncertainty_upper_bound), eps)
    c_un = 1.0 - torch.pow((d_uc / upper).clamp_min(0.0), float(beta))
    c_un = torch.where(d_uc <= upper, c_un.clamp(0.0, 1.0), torch.zeros_like(c_un))
    return prob, c_un, d_uc


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))

    data_cfg = cfg["data"]
    pseudo_cfg = cfg["pseudo"]
    output_dir = ensure_dir(args.output_dir or pseudo_cfg["output_dir"])
    mask_dir = ensure_dir(output_dir / "masks")
    confidence_dir = ensure_dir(output_dir / "confidence")
    uncertainty_dir = ensure_dir(output_dir / "uncertainty")
    uncertainty_map_dir = ensure_dir(output_dir / "uncertainty_map")
    confidence_png_dir = ensure_dir(output_dir / "confidence_png")
    final_threshold = float(args.threshold or pseudo_cfg.get("final_threshold", 0.75))
    selection_metric = str(pseudo_cfg.get("selection_metric", "final_fg_mean"))
    mask_threshold = float(pseudo_cfg.get("mask_threshold", 0.5))
    uncertainty_upper_bound = float(
        pseudo_cfg.get("uncertainty_upper_bound", pseudo_cfg.get("uncertainty_threshold", 0.75))
    )
    uncertainty_beta = float(pseudo_cfg.get("uncertainty_beta", 1.0))
    mag_lambda = float(pseudo_cfg.get("mag_lambda", 0.5))
    kmeans_max_iter = int(pseudo_cfg.get("kmeans_max_iter", 100))
    prior_tolerance_sigma = float(pseudo_cfg.get("prior_tolerance_sigma", 3.0))
    perturbation = pseudo_cfg.get("perturbation", "hflip")
    num_classes = int(data_cfg.get("num_classes", 2))

    labeled_masks = load_labeled_binary_masks(data_cfg)
    if labeled_masks is not None:
        memory_bank = ThicknessMemoryBank.fit_masks(
            labeled_masks,
            n_components=int(pseudo_cfg.get("pca_components", 2)),
            n_clusters=int(pseudo_cfg.get("prior_clusters", 3)),
            random_state=int(cfg.get("seed", 42)),
            kmeans_max_iter=kmeans_max_iter,
            tolerance_sigma=prior_tolerance_sigma,
        )
    else:
        memory_bank = ThicknessMemoryBank.fit(
            get_labeled_mask_files(data_cfg),
            image_size=data_cfg.get("image_size", (512, 512)),
            n_components=int(pseudo_cfg.get("pca_components", 2)),
            n_clusters=int(pseudo_cfg.get("prior_clusters", 3)),
            random_state=int(cfg.get("seed", 42)),
            kmeans_max_iter=kmeans_max_iter,
            tolerance_sigma=prior_tolerance_sigma,
        )
    memory_bank.save(output_dir / "thickness_memory_bank.npz")

    dataset = build_unlabeled_dataset(data_cfg)
    if len(dataset) == 0:
        write_csv(
            output_dir / "pseudo_index.csv",
            [],
            fieldnames=[
                "image_file",
                "pseudo_mask_file",
                "confidence_file",
                "uncertainty_file",
                "uncertainty_map_file",
                "uncertainty_confidence",
                "uncertainty_fg_mean",
                "uncertainty_nonzero_mean",
                "uncertainty_nonzero_ratio",
                "uncertainty_d_mean",
                "prior_score",
                "prior_distance",
                "prior_tolerance",
                "mean_confidence",
                "final_fg_mean",
                "final_nonzero_mean",
                "final_nonzero_ratio",
                "foreground_ratio",
                "selection_metric",
                "selection_score",
                "keep",
            ],
        )
        print("no unlabeled images; saved empty pseudo_index.csv")
        return

    loader = DataLoader(
        dataset,
        batch_size=int(pseudo_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else pseudo_cfg.get("num_workers", 0)),
        pin_memory=True,
    )

    print(f"pseudo samples: {len(dataset)}, batches: {len(loader)}, device: {device}")
    model = build_model(cfg).to(device)
    print(f"loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="pseudo"):
            images = batch["image"].to(device, non_blocking=True)
            prob, uc_confidence, d_uc = uncertainty_confidence(
                model,
                images,
                num_classes=num_classes,
                perturbation=perturbation,
                uncertainty_upper_bound=uncertainty_upper_bound,
                beta=uncertainty_beta,
            )
            pred = prob >= mask_threshold

            pred_np = pred.long().cpu().numpy()
            uc_np = uc_confidence.cpu().numpy()
            d_uc_np = d_uc.cpu().numpy()
            for i, image_file in enumerate(batch["image_file"]):
                stem = Path(image_file).stem
                pseudo_mask_file = mask_dir / f"{stem}.png"
                confidence_file = confidence_dir / f"{stem}.npy"
                uncertainty_file = uncertainty_dir / f"{stem}.npy"
                uncertainty_map_file = uncertainty_map_dir / f"{stem}.npy"
                confidence_png_file = confidence_png_dir / f"{stem}.png"

                prior_stats = memory_bank.score_mask(pred_np[i].astype(np.float32))
                prior_score = float(prior_stats["score"])
                if prior_score > 0:
                    final_confidence = np.where(
                        uc_np[i] > 0,
                        mag_lambda * uc_np[i] + (1.0 - mag_lambda) * prior_score,
                        0.0,
                    )
                else:
                    final_confidence = np.zeros_like(uc_np[i], dtype=np.float32)

                save_mask(pred_np[i], pseudo_mask_file, num_classes=num_classes)
                np.save(confidence_file, final_confidence.astype(np.float32))
                np.save(uncertainty_file, uc_np[i].astype(np.float32))
                np.save(uncertainty_map_file, d_uc_np[i].astype(np.float32))
                if bool(pseudo_cfg.get("save_confidence_png", True)):
                    save_confidence_png(final_confidence, confidence_png_file)

                foreground = pred_np[i].astype(bool)
                uc_mean = float(uc_np[i].mean())
                uc_fg_mean = float(uc_np[i][foreground].mean()) if foreground.any() else 0.0
                uc_nonzero = uc_np[i] > 0
                uc_nonzero_mean = float(uc_np[i][uc_nonzero].mean()) if uc_nonzero.any() else 0.0
                uc_nonzero_ratio = float(uc_nonzero.mean())
                d_mean = float(d_uc_np[i].mean())
                mean_confidence = float(final_confidence.mean())
                final_fg_mean = (
                    float(final_confidence[foreground].mean()) if foreground.any() else 0.0
                )
                final_nonzero = final_confidence > 0
                final_nonzero_mean = (
                    float(final_confidence[final_nonzero].mean()) if final_nonzero.any() else 0.0
                )
                final_nonzero_ratio = float(final_nonzero.mean())
                foreground_ratio = float(foreground.mean())
                selection_scores = {
                    "mean_confidence": mean_confidence,
                    "final_fg_mean": final_fg_mean,
                    "final_nonzero_mean": final_nonzero_mean,
                    "uncertainty_fg_mean": uc_fg_mean,
                    "prior_score": prior_score,
                }
                if selection_metric not in selection_scores:
                    valid = ", ".join(sorted(selection_scores))
                    raise ValueError(
                        f"Unsupported pseudo.selection_metric={selection_metric}. "
                        f"Choose one of: {valid}"
                    )
                selection_score = float(selection_scores[selection_metric])
                rows.append(
                    {
                        "image_file": image_file,
                        "pseudo_mask_file": str(pseudo_mask_file),
                        "confidence_file": str(confidence_file),
                        "uncertainty_file": str(uncertainty_file),
                        "uncertainty_map_file": str(uncertainty_map_file),
                        "uncertainty_confidence": f"{uc_mean:.6f}",
                        "uncertainty_fg_mean": f"{uc_fg_mean:.6f}",
                        "uncertainty_nonzero_mean": f"{uc_nonzero_mean:.6f}",
                        "uncertainty_nonzero_ratio": f"{uc_nonzero_ratio:.6f}",
                        "uncertainty_d_mean": f"{d_mean:.6f}",
                        "prior_score": f"{prior_score:.6f}",
                        "prior_distance": f"{prior_stats['distance']:.6f}",
                        "prior_tolerance": f"{prior_stats['tolerance']:.6f}",
                        "mean_confidence": f"{mean_confidence:.6f}",
                        "final_fg_mean": f"{final_fg_mean:.6f}",
                        "final_nonzero_mean": f"{final_nonzero_mean:.6f}",
                        "final_nonzero_ratio": f"{final_nonzero_ratio:.6f}",
                        "foreground_ratio": f"{foreground_ratio:.6f}",
                        "selection_metric": selection_metric,
                        "selection_score": f"{selection_score:.6f}",
                        "keep": int(selection_score >= final_threshold),
                    }
                )

    write_csv(
        output_dir / "pseudo_index.csv",
        rows,
        fieldnames=[
            "image_file",
            "pseudo_mask_file",
            "confidence_file",
            "uncertainty_file",
            "uncertainty_map_file",
            "uncertainty_confidence",
            "uncertainty_fg_mean",
            "uncertainty_nonzero_mean",
            "uncertainty_nonzero_ratio",
            "uncertainty_d_mean",
            "prior_score",
            "prior_distance",
            "prior_tolerance",
            "mean_confidence",
            "final_fg_mean",
            "final_nonzero_mean",
            "final_nonzero_ratio",
            "foreground_ratio",
            "selection_metric",
            "selection_score",
            "keep",
        ],
    )
    print(f"saved pseudo index: {output_dir / 'pseudo_index.csv'}")


if __name__ == "__main__":
    main()
