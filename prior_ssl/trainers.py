from contextlib import nullcontext

import torch
from tqdm import tqdm

from .losses import segmentation_loss
from .metrics import dice_score, mean_iou, pixel_accuracy
from .utils import AverageMeter
from .visualization import make_segmentation_grid


def autocast_context(device, enabled):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    num_classes,
    amp=False,
    pseudo_weight=1.0,
    ce_weight=1.0,
    dice_weight=1.0,
    log_interval=20,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    loss_meter = AverageMeter()

    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        confidence = batch.get("confidence")
        if confidence is not None:
            confidence = confidence.to(device, non_blocking=True)
            is_pseudo = batch.get("is_pseudo")
            if is_pseudo is not None:
                is_pseudo = is_pseudo.to(device, non_blocking=True).bool()
                sample_weight = torch.where(
                    is_pseudo,
                    torch.full_like(is_pseudo, float(pseudo_weight), dtype=torch.float32),
                    torch.ones_like(is_pseudo, dtype=torch.float32),
                )
                confidence = confidence * sample_weight.view(-1, 1, 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = segmentation_loss(
                logits,
                masks,
                num_classes=num_classes,
                confidence=confidence,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg}


@torch.no_grad()
def validate(model, loader, device, num_classes, ce_weight=1.0, dice_weight=1.0):
    model.eval()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    acc_meter = AverageMeter()
    dsc_meter = AverageMeter()
    visual_grid = None

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = segmentation_loss(
            logits,
            masks,
            num_classes=num_classes,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
        )
        miou = mean_iou(logits, masks, num_classes=num_classes)
        acc = pixel_accuracy(logits, masks)
        dsc = dice_score(logits, masks, num_classes=num_classes)
        if visual_grid is None:
            visual_grid = make_segmentation_grid(
                images,
                masks,
                logits,
                num_classes=num_classes,
            )

        n = images.size(0)
        loss_meter.update(loss.item(), n)
        miou_meter.update(miou.item(), n)
        acc_meter.update(acc.item(), n)
        dsc_meter.update(dsc.item(), n)

    return {
        "loss": loss_meter.avg,
        "mean_iou": miou_meter.avg,
        "pixel_acc": acc_meter.avg,
        "dsc": dsc_meter.avg,
        "visualization": visual_grid,
    }
