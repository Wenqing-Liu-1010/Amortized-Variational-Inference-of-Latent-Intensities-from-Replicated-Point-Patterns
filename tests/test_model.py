import sys
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "source"))
import model as experiment
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heldout import heldout_log_likelihood


def main():
    torch.set_num_threads(1)
    args = experiment.ModelConfig()
    args.latent_dim, args.context_dim, args.hidden_dim = 8, 8, 16
    args.num_hidden_layers = 2
    model = experiment.build_model(
        args, {"feature_mean": torch.zeros(1), "feature_std": torch.ones(1)}, "cpu")
    events = torch.tensor([[[[1.0], [3.0]], [[2.0], [4.0]],
                            [[0.0], [0.0]], [[5.0], [0.0]]]])
    masks = torch.tensor([[[1, 1], [1, 1], [0, 0], [1, 0]]])
    grid = torch.linspace(0, 10, 11).view(1, -1, 1)
    context = model.encode_context(events, masks)
    mu, log_var = model.latent(context)
    order = torch.tensor([2, 0, 3, 1])
    permuted = model.latent(model.encode_context(events[:, order], masks[:, order]))
    torch.testing.assert_close(mu, permuted[0])
    torch.testing.assert_close(log_var, permuted[1])
    for k in (1, 2, 4):
        mean, variance = model.latent(model.encode_context(events[:, :k], masks[:, :k]))
        intensity = model.decode(mean, grid)
        assert torch.isfinite(variance).all() and (intensity > 0).all()
    empty = model.latent(model.encode_context(events[:, 2:3], masks[:, 2:3]))
    assert all(torch.isfinite(value).all() for value in empty)
    context_mu, context_log_var = model.latent(model.encode_context(events[:, :2], masks[:, :2]))
    loss = experiment.neural_process.monte_carlo_nll(
        model, mu=mu, log_var=log_var, target_events=events[:, 2:],
        target_masks=masks[:, 2:], grid_t=grid, t_max=10, mc_samples=2,
        query_chunk_size=4)
    loss = loss + experiment.neural_process.kl_divergence(
        mu, log_var, context_mu, context_log_var)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Unit intensity has log likelihood -T for every held-out sequence.
    class UnitIntensity:
        def eval(self):
            return self

        def encode_context(self, events, masks):
            assert events.shape[1] == 3
            return events

        def latent(self, context):
            return torch.zeros(len(context), 1), torch.zeros(len(context), 1)

        def reparameterize(self, mean, log_var):
            return mean

        def decode(self, latent, times, chunk_size):
            return torch.ones((len(latent), *times.shape[1:-1]))

    hll = heldout_log_likelihood(UnitIntensity(), {"events": events, "masks": masks},
                                grid, args, 1, 3, args.seed)
    assert math.isclose(hll["HLL"], -args.T_max, abs_tol=1e-6)
    assert hll["sequences"] == 4 and hll["events"] == 5
    # Distinguish log E[p] from E[log p] with two constant intensities.
    class TwoIntensities(UnitIntensity):
        draw = 0

        def reparameterize(self, mean, log_var):
            self.draw += 1
            return torch.full_like(mean, 1.0 if self.draw % 2 else 2.0)

        def decode(self, latent, times, chunk_size):
            return latent.reshape(-1, *([1] * (times.ndim - 2))).expand(
                len(latent), *times.shape[1:-1])

    mixture = heldout_log_likelihood(
        TwoIntensities(), {"events": events, "masks": masks}, grid, args, 1, 2, args.seed)
    expected = torch.logsumexp(torch.stack([
        torch.full((4,), -10.0), masks.sum(-1)[0] * math.log(2) - 20
    ]), dim=0).mean() - math.log(2)
    assert math.isclose(mixture["HLL"], expected.item(), abs_tol=2e-6)
    # CUDA autocast can produce bfloat16; quantile requires float32/float64.
    class BfloatDecoder:
        reparameterize = staticmethod(lambda mean, log_var: mean)

        def decode(self, latent, grid, chunk_size):
            return torch.ones(len(latent), grid.shape[1], dtype=torch.bfloat16)

    samples = experiment.neural_process.sample_intensities(
        BfloatDecoder(), mu, log_var, grid, args, 3)
    assert samples.dtype == torch.float32
    assert experiment.neural_process.posterior_interval_width(samples).item() == 0
    print("MODEL_CHECKS_OK")


if __name__ == "__main__":
    main()
