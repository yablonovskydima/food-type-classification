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

## ONNX export and CPU inference (#12)

Export the final trained model using its saved input size and normalization:

```bash
uv run --extra app python -m src.inference.onnx_export_cli --checkpoint checkpoints/resnet18_scratch/best.pt --output-dir checkpoints/onnx_scratch
```

Both `best.pt` (inference metadata) and `last.pt` (training config) are supported.
For older raw state dictionaries or `state_dict` wrappers without preprocessing metadata,
provide `--image-size 224 --normalization food11` explicitly (`imagenet` for the reference).
Overrides that conflict with saved metadata are rejected. Class order must match Food-11.

The output directory must not exist. It is published only after validation succeeds:

- `model.onnx`: self-contained FP32 weights, dynamic batch dimension, fixed input resolution;
  input `images` is normalized NCHW, output `logits` has 11 columns. Class names, resize/crop,
  normalization values, and checkpoint SHA-256 are embedded in the model metadata.
- `export_report.json`: ONNX checker result, PyTorch vs
  ONNX logit/probability comparisons and top-1 agreement at batch sizes 1 and 3, plus CPU timings
  and environment details. Verification uses seeded synthetic tensors, not the test split.

CPU benchmarks use batch size 1, the same intra-op thread count for both backends (default 1),
5 warmup calls and 30 timed calls. Set `--threads`, `--warmup`, or `--iterations` to change these.
Mean, median, p95 latency and throughput cover forward passes only: loading, image decoding,
and preprocessing are excluded. Timings are machine-specific; export parity is not a quality score.

Run the exported model on a photo:

```bash
uv run --extra app python -m src.inference.onnx_predictor --model checkpoints/onnx_scratch/model.onnx --image path/to/dish.jpg --top-k 3
```

The runtime uses the model's embedded preprocessing settings and Pillow/NumPy equivalents of
the project's validation transforms. It does not import PyTorch or torchvision. For a separate
deployment environment, install only `onnxruntime numpy pillow`, copy `model.onnx` and the
`src/inference/__init__.py` / `src/inference/onnx_predictor.py` files preserving their paths,
then run `python -m src.inference.onnx_predictor` from their parent project directory.
The `uv run` commands above use the full development environment, so they still install PyTorch.

Tests exercise real ResNet-18 export with random weights, preprocessing equivalence for varied
image shapes/modes, and inference in a subprocess that blocks PyTorch imports. Exporting the
final trained checkpoint remains a separate step once training is complete.

---

## License & credits

- Code: MIT
- Dataset: Food-11, Multimedia Signal Processing Group (MMSPG), EPFL.
  Singla A., Yuan L., Ebrahimi T. *Food/Non-food Image Classification and Food Categorization using Pre-Trained GoogLeNet Model*, MADiMa 2016.
  Use subject to the dataset's terms.
