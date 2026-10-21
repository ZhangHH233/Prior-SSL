import torch
from torchvision.utils import make_grid


def prediction_from_logits(logits, num_classes=2, threshold=0.5):
    if logits.shape[1] == 1 or num_classes <= 2:
        return (torch.sigmoid(logits[:, 0]) >= threshold).long()
    return torch.argmax(logits, dim=1)


def probability_from_logits(logits, num_classes=2):
    if logits.shape[1] == 1 or num_classes <= 2:
        return torch.sigmoid(logits[:, 0])
    probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values


def to_rgb_image(image):
    if image.shape[0] == 1:
        return image.repeat(3, 1, 1)
    if image.shape[0] >= 3:
        return image[:3]
    return image.mean(dim=0, keepdim=True).repeat(3, 1, 1)


def make_segmentation_grid(images, masks, logits, num_classes=2, max_images=4):
    images = images.detach().cpu()
    masks = masks.detach().cpu().float()
    preds = prediction_from_logits(
        logits.detach().cpu(),
        num_classes=num_classes,
    ).float()

    images = (images * 0.5 + 0.5).clamp(0, 1)
    panels = []
    for idx in range(min(max_images, images.shape[0])):
        image = to_rgb_image(images[idx])
        gt = masks[idx].unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)
        pred = preds[idx].unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)
        overlay = image.clone()
        overlay[0] = torch.maximum(overlay[0], pred[0])
        overlay[1] = torch.maximum(overlay[1], gt[0])
        panels.extend([image, gt, pred, overlay])
    return make_grid(panels, nrow=4, padding=2)
