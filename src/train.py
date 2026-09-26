
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import model as experiment
from heldout import select_checkpoint


def sample_context_target_sizes(generator, total_k):
    """Sample K_C from the paper's two-component discrete-uniform mixture."""
    low = total_k // 2 if int(torch.randint(2, (1,), generator=generator)) == 0 else 1
    context_size = int(torch.randint(low, total_k, (1,), generator=generator))
    return context_size, total_k - context_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-k", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--tasks", type=int, default=0, help="Limit each split for a smoke test; 0 uses all tasks")
    cli = parser.parse_args()
    if min(cli.steps, cli.batch_size, cli.microbatch_size, cli.eval_every) < 1:
        parser.error("Steps and batch sizes must be positive")
    if cli.tasks < 0:
        parser.error("--tasks must be nonnegative")
    if cli.output.exists():
        parser.error(f"Output already exists: {cli.output}")

    data = torch.load(cli.data, map_location="cpu", weights_only=False)
    if data["events"].shape[0] != 5500 or data["events"].shape[1] < cli.total_k:
        raise ValueError("Expected 5500 tasks with at least K_train realizations")

    args = experiment.ModelConfig()
    args.K = cli.total_k
    args.seed = cli.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment.neural_process.set_seed(args.seed)
    model = experiment.build_model(args, data, device)

    cli.output.mkdir(parents=True)
    experiment.write_json(cli.output / "config.json", {
        "data": str(cli.data),
        "seed": cli.seed, "K_train": cli.total_k, "steps": cli.steps,
        "batch_size": cli.batch_size, "microbatch_size": cli.microbatch_size,
        "context_size_distribution": "0.5*Uniform{K/2,...,K-1} + 0.5*Uniform{1,...,K-1}",
        "loss": "NLL(target | q(z | all K)) + KL(q(z | all K) || q(z | context))",
        "checkpoint_selection": "maximum validation held-out HLL at K_train",
        "validation_mc_samples": 300,
        "task_limit": cli.tasks,
    })

    training = {key: data[key][:4500, :cli.total_k][:cli.tasks or None].to(device)
                for key in ("events", "masks")}
    validation = {key: data[key][4500:5000, :cli.total_k][:cli.tasks or None].to(device)
                  for key in ("events", "masks")}
    grid = data["grid"].to(device)
    del data

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.steps)
    task_rng = torch.Generator().manual_seed(args.seed + 8000)
    split_rng = torch.Generator(device=device).manual_seed(args.seed + 9000)
    context_size_counts = {k: 0 for k in range(1, cli.total_k)}
    started = time.monotonic()
    model.train()

    with (cli.output / "training.jsonl").open("w", buffering=1) as train_log:
        for step in tqdm(range(1, cli.steps + 1), mininterval=30):
            context_size, target_size = sample_context_target_sizes(task_rng, cli.total_k)
            context_size_counts[context_size] += 1
            indices = torch.randint(len(training["events"]), (cli.batch_size,), generator=task_rng)
            experiment.neural_process.set_seed(args.seed + step)
            optimizer.zero_grad(set_to_none=True)
            nll_value = kl_value = 0.0

            for first in range(0, cli.batch_size, cli.microbatch_size):
                index = indices[first:first + cli.microbatch_size].to(device)
                events = training["events"][index]
                masks = training["masks"][index]
                width = max(1, int(masks.sum(-1).max().item()))
                events, masks = events[:, :, :width], masks[:, :, :width]
                rows = torch.arange(len(index), device=device)[:, None]
                order = torch.rand(len(index), cli.total_k, device=device, generator=split_rng).argsort(-1)
                target_index = order[:, :target_size]
                context_index = order[:, target_size:]
                weight = len(index) / cli.batch_size

                with experiment.neural_process.autocast_context(device, True, torch.bfloat16):
                    full_context = model.encode_context(events, masks)
                    mu, log_var = model.latent(full_context)
                    context = model.encode_context(events[rows, context_index], masks[rows, context_index])
                    context_mu, context_log_var = model.latent(context)
                    nll = experiment.neural_process.monte_carlo_nll(
                        model, mu=mu, log_var=log_var,
                        target_events=events[rows, target_index],
                        target_masks=masks[rows, target_index], grid_t=grid,
                        t_max=args.T_max, mc_samples=args.nll_mc_samples,
                        query_chunk_size=args.query_chunk_size)
                    kl = experiment.neural_process.kl_divergence(
                        mu, log_var, context_mu, context_log_var)
                    loss = nll + kl
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at update {step}")
                (weight * loss).backward()
                nll_value += weight * float(nll.detach())
                kl_value += weight * float(kl.detach())

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient at update {step}")
            optimizer.step()
            scheduler.step()

            if step == 1 or step % 100 == 0:
                row = {"update": step, "target_size": target_size,
                       "context_size": context_size, "nll": nll_value,
                       "kl": kl_value, "loss": nll_value + kl_value,
                       "grad_norm": float(grad_norm), "seconds": time.monotonic() - started}
                train_log.write(json.dumps(row) + "\n")
                print("TRAIN " + json.dumps(row), flush=True)

            if step % cli.eval_every == 0 or step == cli.steps:
                checkpoint = {
                    "update": step, "args": vars(args).copy(),
                    "model_state_dict": model.state_dict(),
                    "context_size_counts": context_size_counts.copy(),
                }
                torch.save(checkpoint, cli.output / f"checkpoint_{step:05d}.pt")

    training_seconds = time.monotonic() - started
    best = select_checkpoint(model, validation, grid, args, cli.output, 64)
    experiment.write_json(cli.output / "completed.json", {
        "updates": cli.steps, "training_seconds": training_seconds,
        "seconds": time.monotonic() - started, "best_validation_hll": best,
        "context_size_counts": context_size_counts})


if __name__ == "__main__":
    main()
