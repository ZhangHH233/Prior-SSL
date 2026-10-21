from .unet import UNet
from .resunet import ResUNet


def build_model(cfg):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    name = model_cfg.get("name", "resunet").lower()
    num_classes = int(data_cfg.get("num_classes", 2))
    out_channels = 1 if num_classes <= 2 else num_classes

    if name == "resunet":
        return ResUNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=out_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
        )

    if name != "unet":
        raise ValueError(f"Unsupported model: {name}")

    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=out_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
    )


__all__ = ["UNet", "ResUNet", "build_model"]
