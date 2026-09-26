# Amortized Variational Inference of Latent Intensities from Replicated Point Patterns

Minimal PyTorch implementation of the paper's model, training, validation HLL selection, and testing.

## Repository structure

- `src/`: model, training, testing, and shared HLL selection.
- `scripts/`: synthetic and Divvy shell entry points.
- `data/`: provided synthetic datasets and preprocessed Divvy data.
- `checkpoints/`: `synthetic.pt` and `divvy.pt`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Run the following commands from the repository root. Set `PYTHON` to use a different Python executable.

The two synthetic data files are managed with Git LFS. Install Git LFS and run `git lfs install` before adding files to Git; after cloning, run `git lfs pull`. Required files:

```text
data/table3.pt                 # 4500 train + 500 validation + 500 test tasks, 16 sequences
data/test_k64.pt               # the same 500 test tasks, extended to 64 sequences
data/divvy/train.csv.gz
data/divvy/validation.csv.gz
data/divvy/test.csv.gz
checkpoints/synthetic.pt
checkpoints/divvy.pt
```

Only load trusted `.pt` files. Synthetic training reads events, masks, the grid, and training-only feature statistics; true intensities are used only for evaluation.

## Training

Synthetic data:

```bash
MODE=train TOTAL_K=4 SEED=42 bash scripts/synthetic.sh
```

For the paper, repeat with `TOTAL_K=4,8,16` and `SEED=42,43,44,45`. Each command runs one configuration; results go to `outputs/k4_seed42/` by default.

Divvy:

```bash
MODE=train SEED=42 bash scripts/divvy.sh
```

Repeat Divvy with seeds 42--45; results go to `outputs/divvy_seed42/` by default. Both models use 20,000 updates, batches of 256 tasks, three latent draws for the target likelihood, AdamW (learning rate 0.001, weight decay 0.0001), cosine decay, and a checkpoint every 1,000 updates. The loss is target NLL under the full-task latent distribution plus KL(full-task || context).

After training, all saved checkpoints are evaluated by validation-only leave-one-realization-out HLL with 300 latent draws. HLL uses log-mean-exp of sampled likelihoods, not mean log likelihood. Synthetic validation uses the first `TOTAL_K` sequences; Divvy uses four folds. Highest HLL wins (earliest checkpoint on an exact tie). Neither test tasks nor true intensities enter selection.

Each run writes `best_validation.pt`, `best_validation.json`, and `validation.jsonl`. `OUTPUT` overrides the directory; training refuses to overwrite an existing directory. `STEPS`, `BATCH_SIZE`, `MICROBATCH_SIZE`, and `EVAL_EVERY` override training settings. `TASKS` limits each split for smoke tests only.

## Testing

Evaluate the released synthetic checkpoint with four input realizations:

```bash
MODE=test SEQUENCES=4 bash scripts/synthetic.sh
```

Results are saved to `ours_table2.csv`.

`SEQUENCES` sets the test context size: 1, 2, 4, 8, 16 use `table3.pt`; 32 and 64 automatically use `test_k64.pt`. `DATA`, `TASKS`, `SAMPLES` (default 300), and `BATCH_SIZE` override synthetic test settings. Metrics are task-averaged RMSE of the sampled mean intensity, pointwise C95 (percent), IW95, and inference time.

Evaluate the released Divvy checkpoint:

```bash
MODE=test bash scripts/divvy.sh
```

Results are saved to `divvy_test.json`.

Divvy testing uses four-fold HLL with 300 draws. To test newly trained weights, set `CHECKPOINT=outputs/k4_seed42/best_validation.pt` or `CHECKPOINT=outputs/divvy_seed42/best_validation.pt`. Testing never selects a checkpoint. Set `OUTPUT` to keep separate results for each run. Reduced task/sample counts are smoke tests, not paper results.

## Checks

```bash
python tests/test_model.py
python tests/test_workflows.py
```

These use small fixtures, not research data, to check the model/loss, HLL and selection, both training paths, checkpoint loading, and synthetic/Divvy testing. Full paper-scale training is separate. Auxiliary scripts, plots, logs, duplicate data formats, and run outputs are ignored, not deleted.

## License

The code is released under the [MIT License](LICENSE). Third-party datasets remain subject to their original terms.
