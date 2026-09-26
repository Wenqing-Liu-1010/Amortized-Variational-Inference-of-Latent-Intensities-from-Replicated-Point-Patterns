import math
import json
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import neural_process
from model import write_json


@torch.inference_mode()
def heldout_log_likelihood(model, split, grid, args, batch_size, samples, seed):
    """Leave-one-realization-out HLL, averaged over tasks and held-out sequences."""
    events, masks = split["events"], split["masks"]
    if len(events) == 0 or events.shape[1] < 2 or min(batch_size, samples) < 1:
        raise ValueError("HLL needs nonempty tasks, K >= 2, and positive batch/sample counts")
    neural_process.set_seed(seed)
    model.eval()
    total_loglik = total_events = 0.0
    started = time.perf_counter()
    for held in range(events.shape[1]):
        context_index = [index for index in range(events.shape[1]) if index != held]
        for first in range(0, len(events), batch_size):
            context_events = events[first:first + batch_size, context_index]
            context_masks = masks[first:first + batch_size, context_index]
            target_events = events[first:first + batch_size, held:held + 1]
            target_masks = masks[first:first + batch_size, held:held + 1]
            context_width = max(1, int(context_masks.sum(-1).max()))
            target_width = max(1, int(target_masks.sum(-1).max()))
            context_events, context_masks = context_events[:, :, :context_width], context_masks[:, :, :context_width]
            target_events, target_masks = target_events[:, :, :target_width], target_masks[:, :, :target_width]
            context = model.encode_context(context_events, context_masks)
            mu, log_var = model.latent(context)
            draw_loglik = []
            for _ in range(samples):
                latent = model.reparameterize(mu, log_var)
                at_events = model.decode(latent, target_events, args.query_chunk_size)
                at_grid = model.decode(latent, grid, args.query_chunk_size)
                event_term = (torch.log(at_events) * target_masks).sum((1, 2))
                integral = torch.trapezoid(at_grid, dx=args.T_max / (at_grid.size(1) - 1), dim=1)
                draw_loglik.append(event_term - integral)
            predictive = torch.logsumexp(torch.stack(draw_loglik), dim=0) - math.log(samples)
            total_loglik += float(predictive.sum())
            total_events += int(target_masks.sum())
    sequences = events.shape[1] * len(events)
    return {
        "tasks": len(events), "sequences": sequences, "events": total_events,
        "HLL": total_loglik / sequences,
        "pll_per_event": total_loglik / total_events if total_events else None,
        "seconds_per_task": (time.perf_counter() - started) / len(events),
    }


def log_record(kind, record, stream):
    line = json.dumps(record)
    stream.write(line + "\n")
    tqdm.write(f"{kind} {line}", file=sys.stdout)
    sys.stdout.flush()


def validate_checkpoint(model, split, grid, args, output, batch_size, checkpoint, best):
    """Validate the current checkpoint without changing training mode or RNG state."""
    training, numpy_state = model.training, np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            metrics = heldout_log_likelihood(model, split, grid, args, batch_size, 300, args.seed)
    finally:
        np.random.set_state(numpy_state)
        model.train(training)
    score = metrics["HLL"]
    if not math.isfinite(score):
        raise RuntimeError(f"Nonfinite validation HLL at update {checkpoint['update']}")
    record = {"update": checkpoint["update"], "validation_hll": score,
              "mc_samples": 300, "metrics": metrics}
    with (output / "validation.jsonl").open("a", buffering=1) as log:
        log_record("VALIDATION", record, log)
    if best is None or score > best:
        best = score
        checkpoint["validation_hll"] = score
        torch.save(checkpoint, output / "best_validation.pt.tmp")
        (output / "best_validation.pt.tmp").replace(output / "best_validation.pt")
        write_json(output / "best_validation.json", record)
    return best
