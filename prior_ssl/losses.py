import torch
import torch.nn.functional as F


def _is_binary(logits, num_classes):
    return logits.shape[1] == 1 or int(num_classes) <= 2


def dice_loss(logits, target, num_classes, eps=1e-6):
    if _is_binary(logits, num_classes):
        probs = torch.sigmoid(logits[:, 0])
        target = (target > 0).float()
        dims = (0, 1, 2)
        intersection = torch.sum(probs * target, dims)
        cardinality = torch.sum(probs + target, dims)
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        return 1.0 - dice

    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits,
    target,
    num_classes,
    confidence=None,
    ce_weight=1.0,
    dice_weight=1.0,
):
    if _is_binary(logits, num_classes):
        target_float = (target > 0).float()
        ce = F.binary_cross_entropy_with_logits(
            logits[:, 0],
            target_float,
            reduction="none",
        )
    else:
        ce = F.cross_entropy(logits, target.long(), reduction="none")

    if confidence is not None:
        confidence = confidence.to(logits.device).float()
        ce = ce * confidence
        ce = ce.sum() / confidence.sum().clamp_min(1.0)
    else:
        ce = ce.mean()

    loss = ce_weight * ce
    if dice_weight > 0:
        loss = loss + dice_weight * dice_loss(logits, target, num_classes)
    return loss
