# Amortized Variational Inference of Latent Intensities from Replicated Point Patterns

Minimal PyTorch implementation of the paper's model, training, validation HLL selection, and testing.

## Structure

- `src/`: model, training, testing, and HLL selection.
- `data/`: provided synthetic datasets and preprocessed Divvy data.
- `checkpoints/`: `synthetic.pt` and `divvy.pt`.
- `scripts/`: synthetic and Divvy shell entry points.

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

Repeat Divvy with seeds 42--45. Both models use 20,000 updates, batches of 256 tasks, three latent draws for the target likelihood, and a checkpoint every 1,000 updates.


## Testing

Evaluate the released synthetic checkpoint with four input realizations:

```bash
MODE=test SEQUENCES=4 bash scripts/synthetic.sh
```

Results are saved to `synthetic_results.csv`.

`SEQUENCES` sets the test context size: 1, 2, 4, 8, 16 use `synthetic.pt`; 32 and 64 automatically use `synthetic_test_64.pt`.

Evaluate the released Divvy checkpoint:

```bash
MODE=test bash scripts/divvy.sh
```

Results are saved to `divvy_results.json`.



## License

The code is released under the [MIT License](LICENSE). Third-party datasets remain subject to their original terms.
