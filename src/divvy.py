
import argparse
import csv
import gzip
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import model as experiment
from load_checkpoint import load_checkpoint
from train import sample_context_target_sizes
from heldout import heldout_log_likelihood, log_record, validate_checkpoint

K = 4


def load_split(path, limit=0):
    grouped = defaultdict(dict)
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"task_id", "sequence_index", "event_count", "event_seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {required - set(reader.fieldnames or [])}")
        for row in reader:
            events = np.fromstring(row["event_seconds"], sep=" ", dtype=np.float32) / 8640.0
            if len(events) != int(row["event_count"]):
                raise ValueError(f"event_count mismatch for {row['task_id']}/{row['sequence_index']}")
            if not len(events) or not np.all(np.isfinite(events)) or events.min() < 0 or events.max() > 10:
                raise ValueError(f"Invalid events for {row['task_id']}/{row['sequence_index']}")
            if np.any(events[1:] < events[:-1]):
                raise ValueError(f"Unsorted events for {row['task_id']}/{row['sequence_index']}")
            sequence = int(row["sequence_index"])
            if sequence in grouped[row["task_id"]]:
                raise ValueError(f"Duplicate sequence for {row['task_id']}/{sequence}")
            grouped[row["task_id"]][sequence] = events
    task_ids = sorted(grouped)[:limit or None]
    if not task_ids or any(set(grouped[task]) != set(range(K)) for task in task_ids):
        raise ValueError(f"Every task in {path} must contain sequences 0, 1, 2, 3")
    width = max(len(events) for task in task_ids for events in grouped[task].values())
    all_events = torch.zeros(len(task_ids), K, width, 1)
    all_masks = torch.zeros(len(task_ids), K, width)
    for task_index, task in enumerate(task_ids):
        for sequence in range(K):
            events = torch.from_numpy(grouped[task][sequence])
            all_events[task_index, sequence, :len(events), 0] = events
            all_masks[task_index, sequence, :len(events)] = 1
    return {"task_ids": task_ids, "events": all_events, "masks": all_masks}



def train(cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {name: cli.data_dir / f"{name}.csv.gz" for name in ("train", "validation")}
    splits = {name: load_split(path, cli.tasks) for name, path in paths.items()}
    args = experiment.ModelConfig()
    args.K, args.seed = K, cli.seed
    feature_mean, feature_std = experiment.neural_process.compute_event_feature_stats(
        splits["train"]["events"], splits["train"]["masks"])
    model = experiment.build_model(
        args, {"feature_mean": feature_mean, "feature_std": feature_std}, device)
    experiment.neural_process.set_seed(args.seed)
    for split in splits.values():
        split["events"], split["masks"] = split["events"].to(device), split["masks"].to(device)
    grid = torch.linspace(0, args.T_max, args.grid_size, device=device).view(1, -1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.steps)
    task_rng = torch.Generator().manual_seed(args.seed + 8000)
    split_rng = torch.Generator(device=device).manual_seed(args.seed + 9000)
    cli.output.mkdir(parents=True, exist_ok=False)
    experiment.write_json(cli.output / "config.json", {
        "data_dir": str(cli.data_dir), "seed": cli.seed, "K": K,
        "steps": cli.steps, "batch_size": cli.batch_size,
        "microbatch_size": cli.microbatch_size, "eval_every": cli.eval_every,
        "context_size_distribution": "0.5*Uniform{2,3} + 0.5*Uniform{1,2,3}",
        "checkpoint_selection": "maximum validation four-fold HLL",
        "validation_mc_samples": 300,
    })

    context_size_counts = {k: 0 for k in range(1, K)}
    best = None
    validation_seconds = 0.0
    started = time.monotonic()
    model.train()
    with (cli.output / "training.jsonl").open("w", buffering=1) as training_log:
        for step in tqdm(range(1, cli.steps + 1), mininterval=30):
            context_size, target_size = sample_context_target_sizes(task_rng, K)
            context_size_counts[context_size] += 1
            indices = torch.randint(len(splits["train"]["events"]), (cli.batch_size,), generator=task_rng)
            experiment.neural_process.set_seed(args.seed + step)
            optimizer.zero_grad(set_to_none=True)
            nll_value = kl_value = 0.0
            for first in range(0, cli.batch_size, cli.microbatch_size):
                index = indices[first:first + cli.microbatch_size].to(device)
                events, masks = splits["train"]["events"][index], splits["train"]["masks"][index]
                width = max(1, int(masks.sum(-1).max()))
                events, masks = events[:, :, :width], masks[:, :, :width]
                rows = torch.arange(len(index), device=device)[:, None]
                order = torch.rand(len(index), K, device=device, generator=split_rng).argsort(-1)
                target_index = order[:, :target_size]
                context_index = order[:, target_size:]
                with experiment.neural_process.autocast_context(device, True, torch.bfloat16):
                    full = model.encode_context(events, masks)
                    mu, log_var = model.latent(full)
                    context = model.encode_context(events[rows, context_index], masks[rows, context_index])
                    context_mu, context_log_var = model.latent(context)
                    nll = experiment.neural_process.monte_carlo_nll(
                        model, mu=mu, log_var=log_var,
                        target_events=events[rows, target_index], target_masks=masks[rows, target_index],
                        grid_t=grid, t_max=args.T_max, mc_samples=args.nll_mc_samples,
                        query_chunk_size=args.query_chunk_size)
                    kl = experiment.neural_process.kl_divergence(
                        mu, log_var, context_mu, context_log_var)
                    loss = nll + kl
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at update {step}")
                weight = len(index) / cli.batch_size
                (weight * loss).backward()
                nll_value += weight * float(nll.detach())
                kl_value += weight * float(kl.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Nonfinite gradient at update {step}")
            optimizer.step()
            scheduler.step()
            if step == 1 or step % 100 == 0 or step == cli.steps:
                row = {"update": step, "target_size": target_size,
                       "context_size": context_size, "nll": nll_value,
                       "kl": kl_value, "loss": nll_value + kl_value,
                       "grad_norm": float(grad_norm), "seconds": time.monotonic() - started}
                log_record("TRAIN", row, training_log)
            if step % cli.eval_every == 0 or step == cli.steps:
                checkpoint = {"update": step, "args": vars(args).copy(),
                              "model_state_dict": model.state_dict(),
                              "context_size_counts": context_size_counts.copy()}
                torch.save(checkpoint, cli.output / f"checkpoint_{step:05d}.pt")
                validation_started = time.monotonic()
                best = validate_checkpoint(model, splits["validation"], grid, args, cli.output,
                                           cli.eval_batch_size, checkpoint, best)
                validation_seconds += time.monotonic() - validation_started
    seconds = time.monotonic() - started
    experiment.write_json(cli.output / "completed.json", {
        "updates": cli.steps, "training_seconds": seconds - validation_seconds,
        "validation_seconds": validation_seconds,
        "seconds": seconds, "best_validation_hll": best,
        "context_size_counts": context_size_counts})


def test(cli):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args, checkpoint = load_checkpoint(cli.checkpoint, device)
    split = load_split(cli.data_dir / f"{cli.split}.csv.gz", cli.tasks)
    split["events"], split["masks"] = split["events"].to(device), split["masks"].to(device)
    grid = torch.linspace(0, args.T_max, args.grid_size, device=device).view(1, -1, 1)
    result = {"split": cli.split, "seed": args.seed, "selected_update": checkpoint["update"],
              "mc_samples": cli.mc_samples,
              **heldout_log_likelihood(model, split, grid, args, cli.eval_batch_size,
                                       cli.mc_samples, args.seed)}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    experiment.write_json(cli.output, result)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/divvy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--mc-samples", type=int, default=300)
    parser.add_argument("--tasks", type=int, default=0, help="Limit tasks for a smoke test")
    cli = parser.parse_args()
    if min(cli.steps, cli.batch_size, cli.microbatch_size, cli.eval_batch_size,
           cli.eval_every, cli.mc_samples) < 1 or cli.tasks < 0:
        parser.error("Steps, batch/sample counts must be positive; tasks must be nonnegative")
    if cli.mode == "test" and cli.checkpoint is None:
        parser.error("--checkpoint is required in test mode")
    if cli.mode == "train" and cli.output.exists():
        parser.error(f"Output already exists: {cli.output}")
    train(cli) if cli.mode == "train" else test(cli)


if __name__ == "__main__":
    main()
