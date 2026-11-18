# AGENTS.md

Shared context for AI coding agents (and humans) working on this repository.
Read it before making changes. If something here is out of date, update this file in the same PR.

---

## What we are building

An **image classifier** that predicts which of the **11 food categories** a photo belongs to.
It is a university project; the goal is to show the full ML workflow end to end and to be able to
demonstrate training to the supervisor.

- Task: **single-label image classification**. Not object detection — the dataset has no bounding boxes.
- Dataset: [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) (MMSPG, EPFL).
- Main model: **ResNet-18 from `torchvision`, trained from scratch** (`weights=None`).
- Reference model: the same ResNet-18 with ImageNet weights, used **only for comparison**
  to show what pretraining adds.

## Decisions already made — do not change without discussing with the team

| Decision | Reason |
|----------|--------|
| PyTorch + torchvision | `resnet18` is available out of the box; we write our own training loop so every step is visible. |
| ResNet-18, from scratch | Small enough (~11M params) to train on ~10k images; residual blocks + BatchNorm make training from random init stable. Bigger models / ViT would overfit. |
| No YOLO, no timm, no TensorFlow/Keras | Out of scope; keep one stack. |
| Classification only | Food-11 has one label per image and no boxes. |
| Train on `data/processed`, never on `data/raw` | Processed data is deduplicated and resized (see below). |
| `evaluation` split is the test set, used once per final model | Never tune hyper-parameters or pick checkpoints on it — use `validation`. |
| Macro-F1 reported next to accuracy | Classes are imbalanced (Rice ~280 vs Dessert/Soup ~1,500 training images). |
| Class imbalance handled in training (weighted loss or `WeightedRandomSampler`) | Not by copying files on disk. |
| Augmentation and normalisation happen on the fly in transforms | Not baked into `data/processed`. |

## Data pipeline

```
Kaggle ──download_data.py──▶ data/raw/food11 ──prepare_data.py──▶ data/processed/food11 ──▶ training
```

1. `uv run scripts/download_data.py` — downloads Food-11 into `data/raw/food11/{training,validation,evaluation}/<Class>/`
   and checks the image counts (9,866 / 3,430 / 3,347). Kaggle credentials come from `.env`
   (`KAGGLE_API_TOKEN`, see `.env.example`).
2. `uv run scripts/prepare_data.py` — builds `data/processed/food11/`:
   - **removes duplicates** by pixel hash. Food-11 contains ~630 duplicates, mostly across splits
     (564 test images also appear in training or validation). A duplicate is kept only in the earliest
     split, priority `training > validation > evaluation`;
   - **resizes** so the shorter side is 256 px (aspect ratio kept, no upscaling), JPEG quality 95;
   - **renames classes to snake_case**: `bread, dairy_product, dessert, egg, fried_food, meat,
     noodles_pasta, rice, seafood, soup, vegetable_fruit`.

   Result: 9,856 / 3,374 / 2,783 images. The folder is recreated on every run.

Facts about the raw data (checked): all images are RGB JPEGs, none corrupted, median 512×512,
almost all have a shorter side ≥ 224 px. Total ~1.2 GB raw, ~540 MB processed.

## Training setup (planned defaults)

- Input size 128 or 224 px; `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`,
  `RandAugment`, `RandomErasing` for training; `Resize` + `CenterCrop` for validation/test.
- Normalisation with mean/std computed over all pixels of the processed training split
  (`FOOD11_MEAN` / `FOOD11_STD` in `src/data/constants.py`, recompute with
  `uv run scripts/compute_stats.py` after rebuilding `data/processed`). The pretrained
  reference model uses ImageNet statistics instead.
- SGD (momentum 0.9) or AdamW, OneCycle / cosine LR schedule, 60–100 epochs for the scratch model.
- Label smoothing 0.1, weight decay 5e-4, mixed precision (AMP). Early stopping on validation
  macro-F1 is optional (`early_stopping_patience`, `null` disables it): off for the OneCycle
  scratch run, which must reach its final low-LR epochs; on for the cosine pretrained run.
  `best.pt` always keeps the best epoch either way.
- Class imbalance: `imbalance` option of `create_dataloaders` / `build_criterion`
  (`src/data/class_balance.py`, `src/training/losses.py`). Default `weighted_loss`
  (inverse-frequency class weights from the training split); `weighted_sampler` balances batches
  instead. Never use both at once — pass the same strategy to both functions.
- Experiment settings live in YAML configs (`configs/*.yaml`, loaded into `TrainConfig` in
  `src/training/config.py`; unknown keys are rejected), not in code. Fix all random seeds.
- Train with `uv run scripts/train.py [--config configs/<experiment>.yaml] [--resume]`: checks
  data and GPU, trains via `src.training.cli`'s `fit()`, then evaluates `best.pt` on the
  validation split into `reports/<experiment>/validation/` (never the test split).
  Mixed precision uses bfloat16 when the GPU supports it (float16 + GradScaler otherwise);
  validation runs in float32. DataLoader workers are persistent (Windows restarts are slow).
  Validation runs after every epoch; `checkpoints/<experiment>/best.pt` (weights + metadata:
  image size, normalisation, epoch, val metrics) is the model for evaluation/inference,
  `last.pt` is the full state for `--resume`, `history.json` holds per-epoch metrics.
  Resuming requires the same config; an existing run is never overwritten.
  `summary.json` records best epoch/metrics, total training time, precision and hardware.
- On Windows `uv` installs the CUDA 13.2 build of PyTorch (`[tool.uv.sources]` in
  `pyproject.toml`); Linux/macOS use the PyPI wheels.
- Logging to TensorBoard (`runs/<experiment>`), checkpoints to `checkpoints/` — both git-ignored.

## Using trained models

- All checkpoint loading goes through `src/inference/checkpoints.py` (`load_checkpoint`,
  `resolve_preprocessing`). Input size and normalisation are read from the checkpoint metadata,
  so evaluation, the inference CLI, the API, the demo and ONNX export never need them as
  arguments; explicit values are only for checkpoints without metadata and must not conflict.
- The default model is `checkpoints/resnet18_scratch/best.pt` (`DEFAULT_CHECKPOINT`). The CLI
  and API require a trained checkpoint and never fall back to random or ImageNet-only weights
  (an ImageNet backbone with an untrained 11-class head gives meaningless predictions).
- Uploaded photos (API, demo) are decoded with `src/inference/images.py`
  (`decode_uploaded_image`: format/size checks, EXIF orientation).

## Plan and status

1. ✅ Project setup, folder structure
2. ✅ Dataset download script
3. ✅ Dataset processing script (dedup, resize, rename)
4. ✅ EDA notebook (class distribution, examples, mean/std)
5. ✅ Dataset / transforms / DataLoaders (`src/data`), class-imbalance handling (issue #5);
   comparing weighted loss vs sampler by per-class recall needs the training loop
6. ✅ Model factory: ResNet-18 scratch / pretrained (`src/models`)
7. ✅ Training loop with configs, AMP, checkpoints, TensorBoard (`src/training`, issue #7)
8. ⬜ Train ResNet-18 from scratch — **main experiment** (pipeline: `scripts/train.py`, issue #8).
   First run (RTX 3050, 17 min): val macro-F1 0.720 / accuracy 0.710 at epoch 29, but early
   stopping ended it at epoch 44 of 80 while the OneCycle LR was still ~0.07, before the
   annealing phase. Early stopping is now disabled for the scratch config; a full 80-epoch
   rerun (~32 min) is pending, then fill the README results
9. ⬜ Train pretrained reference, compare curves
10. ⬜ Evaluation on the test split: accuracy, top-3, macro-F1, confusion matrix (`src/evaluation`)
11. ⬜ Error analysis, Grad-CAM
12. ⬜ Inference + API / demo (`src/inference`, `app/api`, `app/ui`): CLI and FastAPI checked
    with the first trained checkpoint; Streamlit demo (issue #22) still to be tried by hand
13. ✅ ONNX export, parity checks, CPU benchmarks and PyTorch-free inference (issue #12,
    `src/inference/onnx_*`); final-model export awaits a trained checkpoint
14. ✅ Docker CPU images for API and Streamlit, Compose, healthchecks and real-container
    smoke tests (issue #14); the trained `best.pt` is mounted read-only at runtime;
    still to be run with the final trained checkpoint

Tasks are tracked as GitHub issues; reference the issue in the PR.

## Repository layout

```
.github/workflows/   CI
configs/             experiment configs (YAML)
app/api/             inference API (FastAPI) — depends on src, never the other way round
app/ui/              Streamlit demo using the local inference engine (issue #22)
notebooks/           EDA (one committed notebook with outputs; reusable code goes to src/)
docker/              container healthcheck + smoke test (Dockerfile, compose.yaml at the root;
                     images install CPU torch at the uv.lock versions, see the Dockerfile)
scripts/             one-command utilities: download_data.py, prepare_data.py, compute_stats.py,
                     train.py
src/
  data/              datasets, transforms, dataloaders (training + validation only), class balance
  models/            model factory
  training/          training loop, losses, optimizers, schedulers, checkpoints
  evaluation/        metrics, reports, confusion matrix, Grad-CAM
  inference/         checkpoint loading, predictor + CLI, upload decoding, ONNX export/runtime
tests/unit, tests/integration
data/ checkpoints/ runs/ reports/   generated, git-ignored
```

Rule of thumb: ML logic goes into `src/`; `app/` and `scripts/` only call it.

## Tooling and conventions

- Python **3.13**, dependencies managed with **uv** (`pyproject.toml` + `uv.lock`).
  Run everything with `uv run ...`; add packages with `uv add <pkg>` so the lock file is updated.
- Lint/format with **ruff** (line length 100). Run `uv run ruff check .` before committing.
- Tests with **pytest** in `tests/`. Keep unit tests fast and independent of the real dataset.
- Code, comments, docstrings, commit messages and docs in **English**.
- Prefer small, readable functions and type hints; no notebooks-only logic — reusable code goes to `src/`.

## Git workflow

- `main` — stable, only updated from `dev`.
- `dev` — integration branch.
- Feature work happens on a separate branch per task and is merged into `dev` through a pull request.
- For each task, commit validated changes, push the feature branch, and open a PR targeting `dev`.
  If committing, pushing, or opening the PR is not possible, tell the user which step failed and why.
- Keep PR descriptions brief: describe the change, validation, and related issue.
- Never commit: `.env`, Kaggle tokens, anything in `data/`, `checkpoints/`, `runs/`, model weights (`*.pt`, `*.onnx`).

## Notes for agents

- Do not change the decisions table above on your own; propose it in the PR description instead.
- Do not download the dataset again or delete `data/` unless asked — use `--verify-only` to check it.
- Do not touch `uv.lock` by hand.
- Long-running commands (training) should be started by a human; agents may run short smoke tests
  (one epoch on a small subset).
- When you finish a task, update the **Plan and status** section if it changed.
