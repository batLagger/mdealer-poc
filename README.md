## Embedding Bridge hands project

# Single-hand similarity scripts

Running the commands below will create these three standalone command-line examples of creating a training set, training it, and running the model.

## Setup

Python 3.10+ and run from this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pipeline uses PyTorch, NumPy, Matplotlib, and pandas (imported by the existing
embedding module). scikit-learn is only needed for the unrelated t-SNE plotting
function; pytest is only needed for the existing pair-model tests.
Examples default to CPU. Training can use `--device cuda` when supported by
your PyTorch installation. Training and similarity use Matplotlib's headless
`Agg` backend, so no graphical display or plot interaction is needed.


Pipeline has 3 main steps:
- Generating a hand catalog (500K default, the more the merrier)
- Training the model (creates labels, and saves the output model)
- Run Similarity tests for query hands

## 1. Create the searchable hands dataset

```bash
python create_dataset.py --num-hands 500000 \
  --output hands.pt
```

[create_dataset.py](./create_dataset.py) calls
`generate_training_set`, checks it using `verify_generated_hands`, and saves a CPU
float32 tensor of shape `[N, 4, 13]`. Suit rows are S.H.D.C;  
Catalog creates hands for the training module. 
Random hands are not deduplicated.
The `--output` argument is a filename saved under `datasets/`; default `hands.pt`.

Model and dataset outputs live in separate directories. Choose a different output path for a new dataset or model run.
Relative paths are resolved from the directory where you run the command.


## 2. Train an Embedding similarity model

```bash
python hands_embed_train.py --dataset hands_mid --epochs 19 --modelname model_test5
```

[hands_embed_train.py](./hands_embed_train
.py) calls `train_ltc_fuzzy` with the original
default architecture. The function uses saved hands dataset, choose --dataset for name. Dataset is in datasets folder.
Training loads dataset, labels them, using 80%-20% train/val split,
and prints per-epoch metrics, running it inside the selected output directory. The working directory is restored afterward.

Output: `models/model.pth` (model state dictionary). Use model name in arguments.
The unchanged trainer still calls `plt.show()`;
Matplotlib may print `FigureCanvasAgg is non-interactive` warning,
then training continues and saves the checkpoint.
The small CLI defaults demonstrate the pipeline; they are not an accuracy guarantee. To have a more powerful model, pass
`--num-hands 500000+ --epochs 16+`. reproducibility across
hardware or PyTorch versions is not guaranteed.
Current network is pretty basic (more updates to come), it is still a work in progress. Not using BatchNorm or Dropout, nor patience check.

## 3. Score a sample hand and print the matches

```bash 
 python hands_similarity.py --dataset hands_large \
 --model model_test5  --hand 'AKQ54.A53.9.KQJ9' \
 --losers 1.0 --shape 0.05 --generic-shape 0.1 --top 10

```

[hands_similarity.py](./hands_similarity.py) reuses `create_query_pair` and
`decode_pair` to encode the query, loads the fuzzy model and catalog, calls
`hands_similarity_fuzzy`, and prints the query and top matches using `print_hand`.
Each result includes its rank, zero-based catalog index, similarity score, and
four suit rows in S/H/D/C order. 

Hand notation is `spades.hearts.diamonds.clubs`
Use uppercase `AKQJT98765432`, `T` for ten, and an empty suit for a void, for
example `'AK732..A875.KQJT'`.

Adjust the existing scoring weights:

```bash
 
# Card/Losers embedding only
python hands_similarity.py --dataset hands_large \
    --model model_test5  --hand 'AKQ54.A53.9.KQJ9' \
    --losers 1.0 --shape 0.0 --generic-shape 0.1 --top 10
 
# Exact suit lengths only
python hands_similarity.py --dataset hands_large \
    --model model_test5  --hand 'AKQ54.A53.9.KQJ9' \
    --losers 1.0 --shape 0.9 --generic-shape 0.01 --top 10


# Different hand strengths, generic shape only
python hands_similarity.py --dataset hands_large \
    --model model_test5  --hand 'AKQ54.A53.9.KQJ9' \
    --losers 0.1 --shape 0.1 --generic-shape 1.0 --top 10

```

Weights using decimal values from 0-1.  They should be positive with at
least one positive value. The function combines cosine similarities from the
card/Losers, shape, and generic-shape embedding slices, then min-max normalizes the combined scores over the supplied catalog. Scores are relative to the catalog, not probabilities or absolute measures. A larger hands catalog is obviously better. 

Using base model/dataset, for the `HandLTCfuzzy` architecture (using all relevant aux headers but no Dropout)

## Pipeline summary

```text
generate_training_set -> verify_generated_hands -> separate catalog .pt

generate_training_set -> calc_losers labels -> train_ltc_fuzzy -> model .pth

sample S.H.D.C -> create_query_pair / decode_pair -> query tensor

query + catalog + load_ltc_fuzzy_model -> hands_similarity_fuzzy

sorted indices + scores -> terminal output with print_hand
```

Each script supports `--help`. 

## Verification

For the basic verification test, I ran commands with Python 3.12.3 and PyTorch 2.13.0:
5,000 training hands for five epochs, a saved/reloaded 10,000-hand catalog, and
ten printed matches. Rankings and printed scores were checked against direct
`hands_similarity_fuzzy`.
The generated hands set and model are saved in a text file.. 
You can run similarity immediately; use different output paths
to repeat training or dataset generation without triggering overwrite protection.
