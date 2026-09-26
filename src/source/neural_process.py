import math
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mlp import MLP


class MeanPooling(nn.Module):
    def forward(self, encoded, padding_mask):
        valid = (~padding_mask.bool()).to(encoded.dtype).unsqueeze(-1)
        return (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)


class ReplicateAggregation(nn.Module):
    def __init__(self, context_dim, latent_dim, min_log_var, max_log_var):
        super().__init__()
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.event_pool = MeanPooling()
        self.empty_embedding = nn.Parameter(torch.zeros(context_dim))
        input_dim = 2 * context_dim + 3
        self.mu_proj = nn.Linear(input_dim, latent_dim)
        self.log_var_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Linear(128, latent_dim)
        )
        ratio = -self.min_log_var / (self.max_log_var - self.min_log_var)
        nn.init.constant_(self.log_var_proj[-1].bias, math.log(ratio / (1.0 - ratio)))

    def bounded_log_var(self, value):
        return self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(value)

    def latent(self, encoded_events, masks):
        batch, replicates, event_count, dimension = encoded_events.shape
        pooled = self.event_pool(
            encoded_events.reshape(batch * replicates, event_count, dimension),
            masks.reshape(batch * replicates, event_count) <= 0,
        ).reshape(batch, replicates, dimension)
        pooled = torch.where(
            (masks.sum(-1) == 0)[..., None], self.empty_embedding, pooled
        )
        augmented = torch.cat(
            [pooled.float(), torch.log1p(masks.float().sum(-1, keepdim=True))], dim=-1
        )
        mean = augmented.mean(1)
        variance = (augmented - mean[:, None]).square().sum(1) / replicates
        log_k = mean.new_full((batch, 1), math.log(replicates))
        representation = torch.cat([mean, variance, log_k], dim=-1)
        return (
            self.mu_proj(representation).float(),
            self.bounded_log_var(self.log_var_proj(representation).float()),
        )


class ReplicatedPointProcessModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        context_dim,
        hidden_dims,
        min_log_var,
        max_log_var,
        feature_mean,
        feature_std,
    ):
        super().__init__()
        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32)[:1]
        feature_std = torch.as_tensor(feature_std, dtype=torch.float32)[:1].clamp_min(1e-6)
        self.register_buffer("feature_mean", feature_mean.view(1, 1, 1))
        self.register_buffer("feature_std", feature_std.view(1, 1, 1))
        self.event_encoder = MLP(1, hidden_dims + [context_dim])
        self.aggregation_head = ReplicateAggregation(
            context_dim=context_dim,
            latent_dim=latent_dim,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
        )
        self.decoder = MLP(latent_dim + 1, hidden_dims + [1])

    def encode_context(self, events, masks):
        features = (events - self.feature_mean) / self.feature_std
        return {"h_hier": self.event_encoder(features), "masks_hier": masks}

    def latent(self, context):
        return self.aggregation_head.latent(context["h_hier"], context["masks_hier"])

    @staticmethod
    def reparameterize(mean, log_variance):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, latent, query_times, query_chunk_size=512):
        batch = latent.size(0)
        if query_times.size(0) == 1:
            query_times = query_times.expand(batch, *query_times.shape[1:])
        output_shape = query_times.shape[:-1]
        query_times = query_times.reshape(batch, -1, 1)
        chunk_size = query_times.size(1) if query_chunk_size <= 0 else query_chunk_size
        outputs = []
        for start in range(0, query_times.size(1), chunk_size):
            times = query_times[:, start : start + chunk_size]
            repeated_latent = latent[:, None].expand(-1, times.size(1), -1)
            outputs.append(F.softplus(self.decoder(torch.cat([repeated_latent, times], -1))).squeeze(-1))
        return torch.cat(outputs, 1).reshape(output_shape)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device, enabled, dtype):
    if enabled and device.type == "cuda" and (dtype != torch.bfloat16 or torch.cuda.is_bf16_supported()):
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def compute_event_feature_stats(events, masks):
    valid = events[masks > 0]
    if not valid.numel():
        return torch.zeros(1), torch.ones(1)
    return valid.mean(0), valid.std(0, unbiased=False).clamp_min(1e-6)




def kl_divergence(mean_a, log_var_a, mean_b, log_var_b):
    value = log_var_b - log_var_a + (torch.exp(log_var_a) + (mean_a - mean_b).square()) * torch.exp(-log_var_b) - 1
    return torch.clamp(0.5 * value.sum(-1), min=0).mean()


def point_process_nll(event_intensity, grid_intensity, masks, t_max):
    event_term = (torch.log(event_intensity) * masks).sum(2)
    integral = torch.trapezoid(grid_intensity, dx=t_max / (grid_intensity.size(1) - 1), dim=1)[:, None]
    return (integral - event_term).sum(1).mean()


def monte_carlo_nll(model, *, mu, log_var, target_events, target_masks, grid_t, t_max, mc_samples, query_chunk_size):
    loss = 0
    for _ in range(mc_samples):
        latent = model.reparameterize(mu, log_var)
        loss += point_process_nll(
            model.decode(latent, target_events, query_chunk_size),
            model.decode(latent, grid_t, query_chunk_size),
            target_masks,
            t_max,
        )
    return loss / mc_samples


def root_mean_squared_error(prediction, target):
    return torch.sqrt((prediction - target).square().mean(-1)).mean()


def posterior_coverage(samples, target, coverage=0.95):
    alpha = (1 - coverage) / 2
    lower, upper = torch.quantile(samples, torch.tensor([alpha, 1 - alpha], device=samples.device), dim=0)
    return ((target >= lower) & (target <= upper)).float().mean(-1).mean()


def posterior_interval_width(samples, coverage=0.95):
    alpha = (1 - coverage) / 2
    lower, upper = torch.quantile(samples, torch.tensor([alpha, 1 - alpha], device=samples.device), dim=0)
    return (upper - lower).mean(-1).mean()


@torch.no_grad()
def sample_intensities(model, mean, log_variance, grid, args, sample_count):
    return torch.stack([
        model.decode(model.reparameterize(mean, log_variance), grid, args.query_chunk_size).float()
        for _ in range(sample_count)
    ])
