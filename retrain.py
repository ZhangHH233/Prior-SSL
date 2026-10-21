import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from prior_ssl.datasets import (
    PseudoLabelDataset,
    read_pseudo_index,
)
from prior_ssl.datasets.protocol import build_supervised_dataset, is_goals_dataset, is_nidek_dataset
from prior_ssl.models import build_model
from prior_ssl.trainers import train_one_epoch, validate
from prior_ssl.utils import (
    ensure_dir,
    load_checkpoint,
    load_config,
    resolve_device,
    save_checkpoint,
    save_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prior-SSL retraining with pseudo labels")
    parser.add_argument("--config", default="configs/prior_ssl.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pseudo-index", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))

    data_cfg = cfg["data"]
    retrain_cfg = cfg["retrain"]
    pseudo_cfg = cfg["pseudo"]
    num_classes = int(data_cfg.get("num_classes", 2))
    output_dir = ensure_dir(args.output_dir or retrain_cfg["output_dir"])
    save_config(cfg, output_dir / "config.yaml")
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(retrain_cfg.get("tensorboard_dir", output_dir / "tensorboard"))

    labeled_set = build_supervised_dataset(data_cfg, subset="labeled")

    pseudo_index = args.pseudo_index or f"{pseudo_cfg['output_dir']}/pseudo_index.csv"
    if Path(pseudo_index).exists():
        pseudo_images, pseudo_masks, confidence_files = read_pseudo_index(
            pseudo_index,
            min_confidence=float(retrain_cfg.get("min_sample_confidence", 0.75)),
        )
    else:
        pseudo_images, pseudo_masks, confidence_files = [], [], []
    if pseudo_images:
        pseudo_set = PseudoLabelDataset(
            pseudo_images,
            pseudo_masks,
            confidence_files,
            image_size=data_cfg.get("image_size", (512, 512)),
            num_classes=num_classes,
            in_channels=int(cfg.get("model", {}).get("in_channels", 3)),
        )
        train_set = ConcatDataset([labeled_set, pseudo_set])
    else:
        train_set = labeled_set
    train_loader = DataLoader(
        train_set,
        batch_size=int(retrain_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(retrain_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    val_loader = None
    if is_goals_dataset(data_cfg) or is_nidek_dataset(data_cfg) or (data_cfg.get("val_images") and data_cfg.get("val_masks")):
        val_set = build_supervised_dataset(data_cfg, subset="val")
        val_loader = DataLoader(
            val_set,
            batch_size=int(retrain_cfg.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(retrain_cfg.get("num_workers", 4)),
            pin_memory=True,
        )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(retrain_cfg.get("lr", 5e-5)),
        weight_decay=float(retrain_cfg.get("weight_decay", 1e-4)),
    )
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device, strict=True)

    best_metric = -1.0
    for epoch in range(1, int(retrain_cfg.get("epochs", 80)) + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_classes=num_classes,
            amp=bool(retrain_cfg.get("amp", True)),
            pseudo_weight=float(retrain_cfg.get("pseudo_weight", 0.5)),
            ce_weight=float(retrain_cfg.get("ce_weight", 1.0)),
            dice_weight=float(retrain_cfg.get("dice_weight", 1.0)),
        )
        metric = -train_stats["loss"]
        val_stats = None
        if val_loader is not None:
            val_stats = validate(
                model,
                val_loader,
                device,
                num_classes=num_classes,
                ce_weight=float(retrain_cfg.get("ce_weight", 1.0)),
                dice_weight=float(retrain_cfg.get("dice_weight", 1.0)),
            )
            metric = val_stats["dsc"]

        print(
            f"epoch={epoch} train_loss={train_stats['loss']:.4f}"
            + (
                f" val_loss={val_stats['loss']:.4f}"
                f" val_dsc={val_stats['dsc']:.4f}"
                f" val_miou={val_stats['mean_iou']:.4f}"
                if val_stats
                else ""
            )
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_stats["loss"], epoch)
            if val_stats is not None:
                writer.add_scalar("loss/val", val_stats["loss"], epoch)
                writer.add_scalar("metrics/val_dsc", val_stats["dsc"], epoch)
                writer.add_scalar("metrics/val_miou", val_stats["mean_iou"], epoch)
                writer.add_scalar("metrics/val_pixel_acc", val_stats["pixel_acc"], epoch)
                if val_stats.get("visualization") is not None:
                    writer.add_image(
                        "val/image_gt_pred_overlay",
                        val_stats["visualization"],
                        epoch,
                    )
            writer.flush()

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, metric, cfg)
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metric, cfg)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
