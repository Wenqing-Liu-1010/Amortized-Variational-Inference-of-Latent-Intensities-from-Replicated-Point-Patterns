# Amortized Variational Inference of Latent Intensities from Replicated Point Patterns

PyTorch implementation of the paper's model, training, validation HLL selection, and testing.

## Repository structure

- `src/`: model, training, testing, and shared HLL selection.
- `scripts/`: synthetic and Divvy shell entry points.
- `data/`: provided synthetic datasets and preprocessed Divvy data.
- `checkpoints/`: `synthetic.pt` and `divvy.pt`.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data and checkpoints

```text
data/synthetic.pt                 # 4500 train + 500 validation + 500 test tasks, 16 sequences
data/synthetic_test_64.pt               # the same 500 test tasks, extended to 64 sequences
data/divvy/train.csv.gz
data/divvy/validation.csv.gz
data/divvy/test.csv.gz
checkpoints/synthetic.pt
checkpoints/divvy.pt
```

## Training

Synthetic data:

```bash
MODE=train TOTAL_K=4 SEED=42 bash scripts/synthetic.sh
```

For the paper, repeat with `TOTAL_K=4,8,16` and `SEED=42,43,44,45`. Each command runs one configuration.

Divvy:

```bash
MODE=train SEED=42 bash scripts/divvy.sh
```

Repeat Divvy with seeds 42--45; results go to `outputs/divvy_seed42/` by default. Both models use 20,000 updates, batches of 256 tasks, three latent draws for the target likelihood, AdamW (learning rate 0.001, weight decay 0.0001), cosine decay, and a checkpoint every 1,000 updates.

After training, all saved checkpoints are evaluated by validation-only leave-one-realization-out HLL with 300 latent draws.


## Testing

Evaluate the released synthetic checkpoint with four input realizations:

```bash
MODE=test SEQUENCES=4 bash scripts/synthetic.sh
```

Results are saved to `synthetic_results.csv`.

`SEQUENCES` sets the test context size: 1, 2, 4, 8, 16 use `synthetic.pt`; 32 and 64 automatically use `synthetic_test_64.pt`. `DATA`, `TASKS`, `SAMPLES` (default 300), and `BATCH_SIZE` override synthetic test settings. Metrics are task-averaged RMSE of the sampled mean intensity, pointwise C95 (percent), IW95, and inference time.

Evaluate the released Divvy checkpoint:

```bash
MODE=test bash scripts/divvy.sh
```

Results are saved to `divvy_results.json`.



## License

The code is released under the [MIT License](LICENSE). Third-party datasets remain subject to their original terms.
