# Food Classifier — Food Type Classification with ResNet-18

This project classifies a food photo into one of **11 food categories**.
The main model is **ResNet-18 trained from scratch**: the architecture comes from `torchvision`,
but the weights start random and are learned entirely on the
[Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) dataset.
The same architecture initialised from ImageNet weights is trained as a reference, to show what
pretraining adds.

---

## Workflow at a glance

```
Kaggle
  │  uv run scripts/download_data.py            1. download + verify
  ▼
data/raw/food11/          16,643 images, with duplicates across splits
  │  uv run scripts/prepare_data.py             2. dedup, resize, rename classes
  ▼
data/processed/food11/    9,856 train / 3,374 validation / 2,783 test
  │  uv run scripts/train.py                    3. train + evaluate on validation
  ▼
checkpoints/<experiment>/best.pt                   + TensorBoard logs in runs/
  │                                                + validation reports in reports/
  ├─ uv run python -m src.evaluation.cli ...    4. final test-split evaluation (once)
  ├─ uv run python -m src.inference.cli ...     5. predict photos from the command line
  ├─ uv run uvicorn app.api.main:app            6. REST API
  ├─ uv run streamlit run app/ui/streamlit_app.py   7. web demo
  ├─ uv run python -m src.inference.onnx_export_cli ... 8. ONNX export for CPU deployment
  └─ docker compose --env-file .docker.env up   9. API + demo in containers
```

`best.pt` stores the input size and normalisation it was trained with, so every step after
training (evaluation, CLI, API, demo, ONNX export) picks the right preprocessing automatically.

---

## Dataset

**Food-11** (MMSPG, EPFL) — 16,643 images, 11 classes, pre-split into three parts:

| Split        | Raw images | After dedup | Used for |
|--------------|------------|-------------|----------|
| `training`   | 9,866      | 9,856       | training |
| `validation` | 3,430      | 3,374       | model selection, early stopping, comparing experiments |
| `evaluation` | 3,347      | 2,783       | final test, used once per final model |

**Classes:** Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood,
Soup, Vegetable/Fruit.

**Things to keep in mind:**
- **Duplicates** — ~630 images appear more than once, mostly across splits (564 test images also
  appear in training or validation). They are removed during preparation, otherwise test scores
  would be inflated.
- **Class imbalance** — the training split ranges from ~280 (Rice) to ~1,500 (Dessert, Soup)
  images. Handled with a class-weighted loss; **macro-F1** is reported next to accuracy.
- **Overlapping categories** (Fried food vs Meat, Dessert vs Dairy product) — expected
  confusions, analysed with a confusion matrix and Grad-CAM.

Why ResNet-18: with ~11M parameters it is small enough to train from scratch on ~10k images,
and its residual connections + BatchNorm make training from random initialisation stable.

---

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13 (uv installs it if needed), and an
NVIDIA GPU for training.

```bash
uv sync --all-extras
```

On Windows this installs the CUDA 13.2 build of PyTorch (configured in `[tool.uv.sources]` in
`pyproject.toml`; it needs a recent NVIDIA driver). Check that the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Create your local env file and set `KAGGLE_API_TOKEN` in it
(https://www.kaggle.com/settings/api → Generate New Token):

```bash
cp .env.example .env
```

---

## Running the workflow

### 1. Download the dataset

```bash
uv run scripts/download_data.py
```

Downloads Food-11 into `data/raw/food11/` and checks the image counts. A valid existing copy is
not downloaded again; use `--force` to re-download or `--verify-only` to only check the files.

### 2. Prepare the data

```bash
uv run scripts/prepare_data.py
```

Builds `data/processed/food11/`: removes duplicate images (a duplicate is kept only in the
earliest split, training > validation > evaluation), resizes every image so its shorter side is
256 px, and renames class folders to `snake_case`. The folder is recreated on every run.

If you change the preparation (e.g. `--size`), recompute the normalisation statistics with
`uv run scripts/compute_stats.py` and update `FOOD11_MEAN` / `FOOD11_STD` in
`src/data/constants.py`.

### 3. Train

Main model, ResNet-18 from scratch (`configs/resnet18_scratch.yaml`):

```bash
uv run scripts/train.py
```

Pretrained reference (`configs/resnet18_pretrained.yaml`):

```bash
uv run scripts/train.py --config configs/resnet18_pretrained.yaml
```

The script checks the data and the GPU, trains with validation after every epoch, keeps the
epoch with the best validation macro-F1, then evaluates that model on the validation split and
prints the numbers for the [Results](#results) table. The scratch model always runs its full
OneCycle schedule (80 epochs, ~32 min on an RTX 3050): most of its improvement comes in the
final low-learning-rate epochs, so early stopping is disabled (`early_stopping_patience: null`).
The pretrained reference stops early after 7 epochs without improvement. Outputs per experiment:

| Path | Content |
|------|---------|
| `checkpoints/<experiment>/best.pt` | best weights + metadata (input size, normalisation, epoch, metrics) |
| `checkpoints/<experiment>/last.pt` | full training state for resuming |
| `checkpoints/<experiment>/history.json` | per-epoch loss, accuracy, macro-F1, learning rate |
| `checkpoints/<experiment>/summary.json` | best epoch and metrics, training time, hardware |
| `runs/<experiment>/` | TensorBoard logs |
| `reports/<experiment>/validation/` | validation metrics, confusion matrices, Grad-CAM |

Follow training live with `uv run tensorboard --logdir runs`. If training stops (Ctrl+C,
crash, reboot), continue from the last finished epoch with `--resume`. An existing run is never
overwritten; to start over, change `experiment` in the config or delete its folders.

On an RTX 3050 one epoch takes ~22 s (the first one ~1.5 min while data workers start).
Without a GPU the script refuses to start unless you pass `--allow-cpu` (training would take days).

### 4. Evaluate on the test split (once per final model)

Validation results are produced by `scripts/train.py`. When a model is final, evaluate it once
on the held-out test split:

```bash
uv run python -m src.evaluation.cli --checkpoint checkpoints/resnet18_scratch/best.pt --split evaluation --output-dir reports/resnet18_scratch/test
```

Never use the test split to compare or tune configurations — use the default
`--split validation` for that. A report folder contains:

- `metrics.json` — accuracy, top-3 accuracy, macro-F1, per-class precision/recall/F1/ROC-AUC,
  confusion matrix, checkpoint SHA-256 and run settings;
- `predictions.csv` / `predictions.npz` — per-image predictions and probabilities;
- `confusion_matrix_counts.png`, `confusion_matrix_normalized.png`, `per_class_metrics.png`;
- `gradcam/` — the most confident mistakes, with Grad-CAM heatmaps for the predicted and the
  true class.

`reports/` is git-ignored; copy figures you want to keep (e.g. for the presentation) to `docs/`.

### 5. Predict from the command line

```bash
uv run python -m src.inference.cli --image path/to/dish.jpg
```

Uses `checkpoints/resnet18_scratch/best.pt` unless `--checkpoint` is given. Also supports
`--dir <folder>`, `--top-k N` and `--json`.

### 6. REST API

```bash
uv run uvicorn app.api.main:app
```

Serves `checkpoints/resnet18_scratch/best.pt` (set `CHECKPOINT_PATH` to use another model).
`GET /health` reports whether a model is loaded; `POST /predict` takes an image upload
(`file`, JPG/PNG) and an optional `top_k` query parameter. Interactive docs:
http://localhost:8000/docs.

### 7. Web demo

```bash
uv run streamlit run app/ui/streamlit_app.py
```

Pick a trained model in the sidebar (all `best.pt` files under `checkpoints/` are listed),
upload a photo and choose **Identify food** to see the top-3/top-5 classes with confidences.
The demo calls the model directly; the API does not need to run.

### 8. Export to ONNX (optional)

```bash
uv run python -m src.inference.onnx_export_cli --checkpoint checkpoints/resnet18_scratch/best.pt --output-dir checkpoints/onnx_scratch
uv run python -m src.inference.onnx_predictor --model checkpoints/onnx_scratch/model.onnx --image path/to/dish.jpg
```

The export embeds class names and preprocessing in `model.onnx`, verifies that ONNX and
PyTorch outputs match, and writes CPU timings to `export_report.json`. The ONNX predictor
needs only `onnxruntime`, `numpy` and `pillow` — no PyTorch.

### 9. Run the API and demo in Docker (optional)

Runs steps 6 and 7 on any machine with Docker Desktop (or Docker Engine with Compose), without
Python on the host. Copy the settings file and point `MODEL_CHECKPOINT` at a trained `best.pt`
(default: `./checkpoints/resnet18_scratch/best.pt`):

```bash
cp .docker.env.example .docker.env
docker compose --env-file .docker.env up --build -d --wait
```

The demo opens at http://localhost:8501 and the API docs at http://localhost:8000/docs.
Both containers mount the checkpoint read-only as `/models/best.pt` and read its input size and
normalisation from the file. Stop them with `docker compose --env-file .docker.env down`;
append `api` or `demo` to `up` to start only one.

The images run on CPU as a non-root user, with CPU-only PyTorch wheels pinned to the versions in
`uv.lock`; data, weights and credentials are never copied into them. If the checkpoint is
missing, the API stays up but reports it in `/health` and answers `/predict` with 503, so its
container is unhealthy and `up --wait` fails.

---

## Results

Validation split, best epoch by macro-F1. Filled in from the output of `scripts/train.py`
(also in `checkpoints/<experiment>/summary.json` and `reports/<experiment>/validation/`).

| Model | Hardware | Training time | Epochs (best) | Val accuracy | Val top-3 | Val macro-F1 |
|-------|----------|---------------|---------------|--------------|-----------|--------------|
| ResNet-18 from scratch | — | — | — | — | — | — |
| ResNet-18 ImageNet-pretrained | — | — | — | — | — | — |

Test-split results are added once both models are final.

---

## Project structure

```
configs/             experiment configs (YAML)
scripts/             one-command steps: download_data, prepare_data, compute_stats, train
src/
  data/              dataset, transforms, dataloaders, class balancing, constants
  models/            ResNet-18 factory (scratch / ImageNet-pretrained)
  training/          config, training loop, losses, optimisers/schedules, checkpoints
  evaluation/        metrics, report figures, Grad-CAM, evaluation CLI
  inference/         checkpoint loading, predictor + CLI, image decoding, ONNX export/runtime
app/
  api/               FastAPI service
  ui/                Streamlit demo
notebooks/           exploratory data analysis
docker/              container healthcheck and smoke test (with Dockerfile, compose.yaml)
tests/unit, tests/integration
data/ checkpoints/ runs/ reports/    generated, git-ignored
```

ML logic lives in `src/`; `scripts/` and `app/` only call it.

---

## Development

```bash
uv run pytest
uv run ruff check .
```

Tests run on synthetic data, so they need neither the dataset nor a GPU. CI runs both on every
pull request. Changes to the images (Dockerfile, `src/`, `app/`, dependencies) are also checked
by building both images and running `python docker/smoke_test.py`, which starts real containers
with random weights; run the same locally with Docker Desktop running:

```bash
docker build --target api -t food-classifier-api:local .
docker build --target demo -t food-classifier-demo:local .
python docker/smoke_test.py
``` Branching, conventions and project decisions are described in
[AGENTS.md](AGENTS.md).

---

## License & credits

- Code: MIT
- Dataset: Food-11, Multimedia Signal Processing Group (MMSPG), EPFL.
  Singla A., Yuan L., Ebrahimi T. *Food/Non-food Image Classification and Food Categorization
  using Pre-Trained GoogLeNet Model*, MADiMa 2016. Use subject to the dataset's terms.
