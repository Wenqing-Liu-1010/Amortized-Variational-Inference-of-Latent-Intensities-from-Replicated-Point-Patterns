
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import neural_process
from load_checkpoint import load_checkpoint


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("data/table3.pt"))
parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/synthetic.pt"))
parser.add_argument("--output", type=Path, default=Path("ours_table2.csv"))
parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
parser.add_argument("--tasks", type=int, default=0, help="Limit each split; 0 uses the full split")
parser.add_argument("--sequences", type=int, default=4, help="Number of input sequences per task")
parser.add_argument("--samples", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=16)
cli = parser.parse_args()

if cli.tasks < 0 or cli.sequences < 1 or cli.samples < 2 or cli.batch_size < 1:
    parser.error("Expected tasks >= 0, sequences/batch-size >= 1, and samples >= 2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torch.load(cli.data, map_location="cpu", weights_only=False)
if cli.sequences > data["events"].shape[1]:
    parser.error(f"Data contain only {data['events'].shape[1]} sequences per task")
model, args, _ = load_checkpoint(cli.checkpoint, device)
args.coverage_samples = cli.samples
args.batch_size = cli.batch_size
grid = data["grid"].to(device)

model.eval()
split_ranges = {"train": (0, 4500), "validation": (4500, 5000), "test": (5000, 5500)}
if len(data["events"]) == 500:
    if cli.split != "test":
        parser.error("The 500-task dataset contains only the test split")
    split_ranges = {"test": (0, 500)}
elif len(data["events"]) != 5500:
    parser.error("Expected the 5500-task dataset or the 500-task test dataset")
selected = split_ranges if cli.split == "all" else {cli.split: split_ranges[cli.split]}
rows = []
with torch.no_grad():
    for split, (first, last) in selected.items():
        last = min(last, first + cli.tasks) if cli.tasks else last
        events = data["events"][first:last, :cli.sequences].to(device)
        masks = data["masks"][first:last, :cli.sequences].to(device)
        truth = data["truth"][first:last].to(device)
        count = last - first
        totals = {"RMSE": 0.0, "C95 (%)": 0.0, "IW95": 0.0, "Time (ms)": 0.0}
        warm_context = model.encode_context(events[:1], masks[:1])
        warm_mu, warm_log_var = model.latent(warm_context)
        neural_process.sample_intensities(model, warm_mu, warm_log_var, grid, args, cli.samples)
        if device.type == "cuda":
            torch.cuda.synchronize()
        neural_process.set_seed(args.seed + 10_000 + cli.sequences)
        for start in range(0, count, cli.batch_size):
            stop = min(start + cli.batch_size, count)
            batch = stop - start
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            with neural_process.autocast_context(device, True, torch.bfloat16):
                context = model.encode_context(events[start:stop], masks[start:stop])
                mu, log_var = model.latent(context)
                samples = neural_process.sample_intensities(model, mu, log_var, grid, args, cli.samples)
            if device.type == "cuda":
                torch.cuda.synchronize()
            totals["Time (ms)"] += 1000 * (time.perf_counter() - tick)
            target = truth[start:stop]
            totals["RMSE"] += neural_process.root_mean_squared_error(samples.mean(0), target).item() * batch
            totals["C95 (%)"] += neural_process.posterior_coverage(samples, target).item() * 100 * batch
            totals["IW95"] += neural_process.posterior_interval_width(samples).item() * batch
        row = {"Split": split, "Tasks": count,
               "Method": f"Ours (K_train={args.K}, K_test={cli.sequences})",
               **{key: f"{value / count:.2f}" for key, value in totals.items()}}
        rows.append(row)
        print(row, flush=True)

cli.output.parent.mkdir(parents=True, exist_ok=True)
with cli.output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
