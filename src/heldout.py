import math
import json
import time

import torch
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


def select_checkpoint(model, split, grid, args, output, batch_size):
    """Select by validation HLL only, after all training checkpoints are saved."""
    best = None
    paths = sorted(output.glob("checkpoint_*.pt"))
    if not paths:
        raise ValueError("No training checkpoints found")
    with (output / "validation.jsonl").open("w", buffering=1) as log:
        for path in paths:
            checkpoint = torch.load(path, map_location=grid.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            metrics = heldout_log_likelihood(model, split, grid, args, batch_size, 300, args.seed)
            score = metrics["HLL"]
            if not math.isfinite(score):
                raise RuntimeError(f"Nonfinite validation HLL: {path.name}")
            record = {"update": checkpoint["update"], "validation_hll": score,
                      "mc_samples": 300, "metrics": metrics}
            log.write(json.dumps(record) + "\n")
            print("VALIDATION " + json.dumps(record), flush=True)
            if best is None or score > best:
                best = score
                checkpoint["validation_hll"] = score
                torch.save(checkpoint, output / "best_validation.pt.tmp")
                (output / "best_validation.pt.tmp").replace(output / "best_validation.pt")
                write_json(output / "best_validation.json", record)
    return best
