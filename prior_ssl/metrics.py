import torch


@torch.no_grad()
def prediction_from_logits(logits, num_classes, threshold=0.5):
    if logits.shape[1] == 1 or int(num_classes) <= 2:
        return (torch.sigmoid(logits[:, 0]) >= threshold).long()
    return torch.argmax(logits, dim=1)


@torch.no_grad()
def mean_iou(logits, target, num_classes, eps=1e-6):
    pred = prediction_from_logits(logits, num_classes)
    target = target.long()
    ious = []
    for cls_idx in range(num_classes):
        pred_cls = pred == cls_idx
        target_cls = target == cls_idx
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append((intersection + eps) / (union + eps))
    if not ious:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(ious).mean()


@torch.no_grad()
def pixel_accuracy(logits, target):
    num_classes = logits.shape[1] if logits.shape[1] > 1 else 2
    pred = prediction_from_logits(logits, num_classes)
    return (pred == target.long()).float().mean()


@torch.no_grad()
def dice_score(logits, target, num_classes=2, eps=1e-6):
    pred = prediction_from_logits(logits, num_classes)
    target = (target > 0).long() if int(num_classes) <= 2 else target.long()

    if int(num_classes) <= 2:
        pred_fg = pred == 1
        target_fg = target == 1
        intersection = (pred_fg & target_fg).sum().float()
        cardinality = pred_fg.sum().float() + target_fg.sum().float()
        return (2.0 * intersection + eps) / (cardinality + eps)

    scores = []
    for cls_idx in range(1, int(num_classes)):
        pred_cls = pred == cls_idx
        target_cls = target == cls_idx
        cardinality = pred_cls.sum().float() + target_cls.sum().float()
        if cardinality > 0:
            intersection = (pred_cls & target_cls).sum().float()
            scores.append((2.0 * intersection + eps) / (cardinality + eps))
    if not scores:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(scores).mean()
