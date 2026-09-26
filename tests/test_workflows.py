"""Small CPU end-to-end checks; no released data or GPU required."""

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "source"))
import divvy
import heldout
import model as experiment
import train
from load_checkpoint import load_checkpoint


def main():
    torch.set_num_threads(1)
    original_config = experiment.ModelConfig

    def small_config():
        args = original_config()
        args.context_dim = args.latent_dim = 4
        args.hidden_dim, args.num_hidden_layers, args.grid_size = 8, 1, 11
        return args

    with tempfile.TemporaryDirectory() as directory, patch.object(experiment, "ModelConfig", small_config), \
            patch("torch.cuda.is_available", return_value=False):
        root = Path(directory)
        data = {"events": torch.ones(5500, 4, 2, 1), "masks": torch.ones(5500, 4, 2),
                "grid": torch.linspace(0, 10, 11).view(1, -1, 1),
                "feature_mean": torch.zeros(1), "feature_std": torch.ones(1)}
        path, output = root / "synthetic.pt", root / "synthetic"
        torch.save(data, path)  # No truth or test_k64: neither is needed for training.
        with patch.object(sys, "argv", ["train", "--data", str(path), "--output", str(output),
                                        "--total-k", "4", "--seed", "42", "--steps", "2",
                                        "--eval-every", "1", "--tasks", "1",
                                        "--batch-size", "2", "--microbatch-size", "1"]):
            train.main()
        rows = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(row["mc_samples"] == 300 for row in rows)
        selected = json.loads((output / "best_validation.json").read_text())
        assert selected["validation_hll"] == max(row["validation_hll"] for row in rows)
        assert not (output / "test.jsonl").exists()
        model, args, _ = load_checkpoint(output / "best_validation.pt")

        # Selection must maximize HLL, not minimize it or use truth/test metrics.
        with patch.object(heldout, "heldout_log_likelihood",
                          side_effect=[{"HLL": -2.0}, {"HLL": -1.0}]) as score:
            assert heldout.select_checkpoint(model, {}, data["grid"], args, output, 1) == -1.0
            assert all(call.args[-2] == 300 for call in score.call_args_list)
        assert load_checkpoint(output / "best_validation.pt")[2]["update"] == 2

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHON=sys.executable, MODE="test", CHECKPOINT=str(output / "best_validation.pt"),
                   DATA=str(path), OUTPUT=str(root / "synthetic.csv"), TASKS="1", SAMPLES="3",
                   SEQUENCES="4", BATCH_SIZE="1")
        data["truth"] = torch.ones(5500, 11)
        torch.save(data, path)
        subprocess.run(["bash", str(ROOT / "scripts" / "synthetic.sh")], env=env, check=True)
        with (root / "synthetic.csv").open() as stream:
            assert next(csv.DictReader(stream))["Tasks"] == "1"
        data["events"] = torch.ones(500, 64, 2, 1)
        data["masks"] = torch.ones(500, 64, 2)
        data["truth"] = torch.ones(500, 11)
        torch.save(data, path)
        env["SEQUENCES"] = "64"
        subprocess.run(["bash", str(ROOT / "scripts" / "synthetic.sh")], env=env, check=True)

        for split in ("train", "validation"):
            with gzip.open(root / f"{split}.csv.gz", "wt", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["task_id", "sequence_index", "event_count", "event_seconds"])
                writer.writerows([[split, k, 2, "8640 17280"] for k in range(4)])
        divvy_output = root / "divvy"
        with patch.object(sys, "argv", ["divvy", "--mode", "train", "--data-dir", str(root),
                                        "--output", str(divvy_output), "--steps", "1",
                                        "--batch-size", "2", "--microbatch-size", "1"]):
            divvy.main()
        assert load_checkpoint(divvy_output / "best_validation.pt")[2]["update"] == 1
        env.update(CHECKPOINT=str(divvy_output / "best_validation.pt"), DATA_DIR=str(root),
                   OUTPUT=str(root / "divvy.json"), MC_SAMPLES="3", SEED="999")
        (root / "test.csv.gz").write_bytes((root / "validation.csv.gz").read_bytes())
        subprocess.run(["bash", str(ROOT / "scripts" / "divvy.sh")], env=env, check=True)
        result = json.loads((root / "divvy.json").read_text())
        assert result["tasks"] == 1 and result["sequences"] == 4 and result["seed"] == 42
    print("WORKFLOW_CHECKS_OK")


if __name__ == "__main__":
    main()
