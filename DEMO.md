# Single-hand fuzzy similarity demo

These three standalone command-line examples call the existing functions without
changing their definitions, model architecture, training, or scoring algorithms.
The only library change removes an unused import of the nonexistent
`build_supervised_1nt_dataloaders`, which previously prevented importing
[hand_emb.py](./hand_emb.py).

## Setup

Use Python 3.10+ and run from this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pipeline uses PyTorch, NumPy, Matplotlib, and pandas (imported by the existing
embedding module). scikit-learn is only needed for the unrelated t-SNE plotting
function; pytest is only needed for the existing pair-model tests.
The examples default to CPU. Training can use `--device cuda` when supported by
your PyTorch installation. Training and similarity use Matplotlib's headless
`Agg` backend, so no graphical display or plot interaction is needed.

## 1. Train a fuzzy model

```bash
python train_example.py --num-hands 5000 --epochs 5 --batch-size 256 \
  --output-dir demo_output/models
```

[train_example.py](./train_example.py) calls `train_ltc_fuzzy` with the original
default architecture. The function generates its own random training hands,
labels them with `calc_losers`, uses its existing 80/20 training/validation split,
and prints per-epoch metrics. Its fixed checkpoint filename is preserved by
running it inside the selected output directory. The working directory is
restored afterward.

Output: `demo_output/models/ltc_fuzzy_model.pth` (model state dictionary).
No plots are displayed or saved. The unchanged trainer still calls `plt.show()`;
Matplotlib may print a harmless `FigureCanvasAgg is non-interactive` warning,
then training continues and saves the checkpoint.
The small CLI defaults demonstrate the pipeline; they are not an accuracy
guarantee. To use the original training budget, pass
`--num-hands 500000 --epochs 14`. `--seed` defaults to 42; reproducibility across
hardware or PyTorch versions is not guaranteed.

## 2. Create the searchable hands dataset

```bash
python create_dataset_example.py --num-hands 10000 \
  --output demo_output/datasets/hands.pt
```

[create_dataset_example.py](./create_dataset_example.py) calls
`generate_training_set`, checks it using `verify_generated_hands`, and saves a CPU
float32 tensor of shape `[10000, 4, 13]`. Each hand has exactly 13 cards, with
binary entries. Suit rows are spades, hearts, diamonds, clubs; rank columns run
from 2 through ace. This catalog is separate from the hands created internally
for training. Random hands are not deduplicated.

Model and dataset outputs live in separate directories. Both scripts refuse to
overwrite existing output files; choose a different output path for another run.
Relative paths are resolved from the directory where you run the command.

## 3. Score a sample hand and print the matches

```bash
python similarity_example.py --hand 'AKQ.J53.9.KQJ985' \
  --model demo_output/models/ltc_fuzzy_model.pth \
  --dataset demo_output/datasets/hands.pt --top 10
```

[similarity_example.py](./similarity_example.py) reuses `create_query_pair` and
`decode_pair` to encode the query, loads the fuzzy model and catalog, calls
`hands_similarity_fuzzy`, and prints the query and top matches using `print_hand`.
Each result includes its rank, zero-based catalog index, similarity score, and
four suit rows in S/H/D/C order. Fewer than `--top` results are printed if the
catalog is smaller.

Hand notation is `spades.hearts.diamonds.clubs`, with exactly 13 distinct cards.
Use uppercase `AKQJT98765432`, `T` for ten, and an empty suit for a void, for
example `'AKQJT98765432...'`.

Adjust the existing scoring weights:

```bash
# Card/LTC embedding only
python similarity_example.py --hand 'AKQ.J53.9.KQJ985' \
  --ltc-param 1 --shape-param 0 --sorted-shape-param 0

# Exact suit lengths only
python similarity_example.py --hand 'AKQ.J53.9.KQJ985' \
  --ltc-param 0 --shape-param 1 --sorted-shape-param 0

# Suit-agnostic sorted shape only
python similarity_example.py --hand 'AKQ.J53.9.KQJ985' \
  --ltc-param 0 --shape-param 0 --sorted-shape-param 1
```

All three weights default to 1. They must be finite and nonnegative, with at
least one positive value. The function combines cosine similarities from the
card/LTC, exact-shape, and sorted-shape embedding slices, then min-max normalizes
the combined scores over the supplied catalog. Scores are relative to that
catalog, not probabilities or absolute measures of bridge strength. Ties are
possible; when every raw score is equal (including a one-hand catalog), the
existing normalization returns zero for every hand.

Use only trusted model/dataset files. The loader expects the original
`HandLTCfuzzy` architecture, not `HandLTC`, `HandLTCfuzzy2`, or pair checkpoints.
The older `hands_similarity0`, `hands_similarity`, and `hands_similarity_new`
functions are not used because they belong to the separate `HandLTC` pipeline.

## Pipeline summary

```text
generate_training_set -> calc_losers labels -> train_ltc_fuzzy -> model .pth
generate_training_set -> verify_generated_hands -> separate catalog .pt
sample S.H.D.C -> create_query_pair / decode_pair -> query tensor
query + catalog + load_ltc_fuzzy_model -> hands_similarity_fuzzy
sorted indices + scores -> terminal output with print_hand
```

Each script supports `--help`. Missing/invalid files and invalid inputs are
reported as errors; the scripts do not silently substitute another model or
dataset.

## Verification

The commands above were run successfully using Python 3.12.3 and PyTorch 2.13.0:
5,000 training hands for five epochs, a saved/reloaded 10,000-hand catalog, and
ten printed matches. Rankings and printed scores were checked against direct
`hands_similarity_fuzzy` calls for equal weights and each of the three slices
individually. Input errors, overwrite protection, void notation, and
single-hand/empty/wrong-type catalogs were also checked.

The generated demo checkpoint and catalog are already present after this
verification. You can run similarity immediately; use different output paths
to repeat training or dataset generation without triggering overwrite protection.
