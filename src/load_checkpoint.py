
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import model as model_code


def load_checkpoint(checkpoint_path, device="cpu"):
    """Return `(model, args, checkpoint)` ready for encoding and decoding."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    data_stats = {
        "feature_mean": state["feature_mean"].cpu(),
        "feature_std": state["feature_std"].cpu(),
    }
    config = model_code.ModelConfig()
    for name, value in checkpoint["args"].items():
        if hasattr(config, name):
            setattr(config, name, value)
    model = model_code.build_model(config, data_stats, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint
