# Food Classifier - Food Type Classification with ResNet-18

This project classifies a food photo into one of **11 food categories**.
The model is **ResNet-18 trained from scratch**: the architecture comes from `torchvision`,
but the weights start random and are learned entirely on the
[Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) dataset.

---

## Dataset

**Food-11** (MMSPG, EPFL) - 16,643 images, 11 classes, pre-split into three parts:

| Split        | Images | Used for |
|--------------|--------|----------|
| `training`   | 9,866  | training |
| `validation` | 3,430  | model selection, early stopping |
| `evaluation` | 3,347  | final test (used once) |

**Classes:** Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit.

**Things to keep in mind:**
- **Class imbalance** - the training split ranges from ~280 (Rice) to ~1,500 (Dessert, Soup) images.
  Handled with class-weighted loss or `WeightedRandomSampler`; **macro-F1** is reported next to accuracy.
- **Variable image sizes** - resized and cropped to a fixed input size (128 or 224 px).
- **Overlapping categories** (Fried food vs Meat, Dessert vs Dairy product) - expected confusions, analysed with a confusion matrix.

Why ResNet-18: with ~11M parameters it is small enough to train from scratch on ~10k images,
and its residual connections + BatchNorm make training from random initialisation stable.
Bigger models would overfit or need far more data.

---

## Getting started

The project uses [uv](https://docs.astral.sh/uv/). `uv run` installs the dependencies on first use.

1. Create your local env file and set `KAGGLE_API_TOKEN` in it
   (https://www.kaggle.com/settings/api → Generate New Token):

   ```bash
   cp .env.example .env
   ```

2. Download the dataset into `data/raw/food11/` and verify it:

   ```bash
   uv run scripts/download_data.py
   ```

   The script skips the download if a valid copy already exists.
   Use `--force` to download again or `--verify-only` to just check the files on disk.

3. Prepare the training data in `data/processed/food11/`:

   ```bash
   uv run scripts/prepare_data.py
   ```

   This removes duplicate images, resizes every image so its shorter side is 256 px
   (use `--size` to change it) and renames class folders to `snake_case`.
   Food-11 contains ~620 duplicates, mostly across splits: 564 test images also appear in
   training or validation. Each duplicate is kept only in the earliest split
   (training > validation > evaluation), so the test split is ~2,780 images instead of 3,347.

---

## Tech stack

- Python 3.12+
- PyTorch + torchvision
- scikit-learn (metrics), matplotlib / seaborn (plots)
- TensorBoard (training curves)
- pytorch-grad-cam (interpretability)
- Gradio (demo), FastAPI (inference API), ONNX Runtime
- Docker

---

## Training (#7)

Each experiment is a YAML file in `configs/`:

- `resnet18_scratch.yaml` — main model, random initialisation, SGD + OneCycle, up to 80 epochs;
- `resnet18_pretrained.yaml` — reference model from ImageNet weights, AdamW + cosine, up to 25 epochs.

```bash
uv run python -m src.training.cli --config configs/resnet18_scratch.yaml
```

After every epoch the model is validated. Results go to `checkpoints/<experiment>/`:

- `best.pt` — weights with the best validation macro-F1 plus metadata (image size,
  normalisation, epoch, metrics); use it for evaluation, inference and the demo;
- `last.pt` — full training state, used to continue an interrupted run;
- `history.json` — per-epoch loss, accuracy, macro-F1 and learning rate.

Training stops early when validation macro-F1 has not improved for `early_stopping_patience`
epochs. An existing run is never overwritten: continue it with

```bash
uv run python -m src.training.cli --config configs/resnet18_scratch.yaml --resume checkpoints/resnet18_scratch/last.pt
```

or change `experiment` in the config to start a new one. Follow training live with:

```bash
uv run tensorboard --logdir runs
```

Useful options: `--device cpu|cuda`, `--num-workers 0` (if data loading fails on Windows),
`--epochs N`, and `--limit-batches N` for a quick smoke test. Full training needs a CUDA GPU:
on a CPU one epoch at 224 px takes tens of minutes.

---

## Roadmap

- [ ] Data pipeline and EDA
- [ ] ResNet-18 trained from scratch
- [ ] Scratch vs pretrained comparison
- [ ] Error analysis, confusion matrix, Grad-CAM
- [ ] ONNX export, inference API, Gradio demo

---

## API container (#14)

Docker Desktop must use Linux containers. This image serves the FastAPI API on CPU;
Streamlit container packaging is a separate follow-up. No local Python installation is
needed to run the API.

1. Copy `.docker.env.example` to `.docker.env` and set `MODEL_CHECKPOINT` to your trained
   checkpoint file (relative to the project root or an absolute host path).
2. Start the service from the repository root:

   ```bash
   docker compose --env-file .docker.env up --build -d --wait
   ```

3. Open `http://localhost:8000/docs` to upload a photo to `/predict`.
   `/health` reports readiness. Stop with:

   ```bash
   docker compose --env-file .docker.env down
   ```

The checkpoint is mounted read-only at `/models/model.pt`; weights, datasets and `.env`
credentials are excluded from the image and build context. A missing checkpoint stops
startup. `best.pt` metadata or `last.pt` config supplies image size and normalization.
For an older checkpoint without metadata, set `IMAGE_SIZE` and `NORMALIZATION` in
`.docker.env` to match training (defaults: 224 and food11). Conflicting saved settings
are rejected. Set `API_PORT` to change the host port. Compose binds to localhost.

The multi-stage image runs as a non-root user and installs only serving dependencies.
Package versions and constraints come from `uv.lock`; torch/torchvision use CPU wheels
from the official PyTorch index at the same version. Training and demo dependencies
are not installed. The API can also run directly:

```bash
docker build --target api -t food-classifier-api:local .
docker run --rm -p 127.0.0.1:8000:8000 --mount type=bind,source=/absolute/path/best.pt,target=/models/model.pt,readonly -e CHECKPOINT_PATH=/models/model.pt food-classifier-api:local
```

For developers, `python docker/smoke_test.py` exercises the built image with a temporary
random-weight checkpoint: readiness, real photo upload, CPU-only PyTorch, non-root runtime,
and failure without a checkpoint. It cleans up its own container and temporary files;
it does not use the real dataset or train a model. CI runs the same build and smoke test.

---

## License & credits

- Code: MIT
- Dataset: Food-11, Multimedia Signal Processing Group (MMSPG), EPFL.
  Singla A., Yuan L., Ebrahimi T. *Food/Non-food Image Classification and Food Categorization using Pre-Trained GoogLeNet Model*, MADiMa 2016.
  Use subject to the dataset's terms.
