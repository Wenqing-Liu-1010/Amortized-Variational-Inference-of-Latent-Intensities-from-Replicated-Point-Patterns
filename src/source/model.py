
import json
from dataclasses import dataclass

import torch

import neural_process


@dataclass
class ModelConfig:
    seed: int = 42
    K: int = 16
    T_max: float = 10.0
    grid_size: int = 300
    latent_dim: int = 256
    context_dim: int = 256
    hidden_dim: int = 512
    num_hidden_layers: int = 3
    min_log_var: float = -6.0
    max_log_var: float = 3.0
    lr: float = 1e-3
    grad_clip: float = 5.0
    query_chunk_size: int = 512
    nll_mc_samples: int = 3
    coverage_samples: int = 300
    batch_size: int = 16


def build_model(args, data, device):
    neural_process.set_seed(args.seed)
    model = neural_process.ReplicatedPointProcessModel(
        context_dim=args.context_dim,
        latent_dim=args.latent_dim,
        hidden_dims=[args.hidden_dim] * args.num_hidden_layers,
        min_log_var=args.min_log_var,
        max_log_var=args.max_log_var,
        feature_mean=data["feature_mean"],
        feature_std=data["feature_std"],
    )
    return model.to(device)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
