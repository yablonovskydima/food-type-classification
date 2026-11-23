# AGENTS.md

Shared context for AI coding agents (and humans) working on this repository.
Read it before making changes. If something here is out of date, update it in the same PR.

## What we are building

An image classifier that predicts which of the **11 food categories** a photo belongs to — a
university project that shows the full ML workflow end to end.

- Task: single-label image classification (no detection: the dataset has no boxes).
- Dataset: [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset).
- Main model: ResNet-18 from `torchvision`, **trained from scratch**.
- Reference model: the same ResNet-18 with ImageNet weights, only for comparison.

## Decisions — do not change without discussing with the team

| Decision | Reason |
|----------|--------|
| PyTorch + torchvision, own training loop | Every training step is visible. |
| ResNet-18 from scratch | ~11M params fits ~10k images; bigger models overfit. |
| No YOLO, timm, TensorFlow/Keras | Out of scope; one stack. |
| Train on `data/processed`, never `data/raw` | Processed data is deduplicated and resized. |
| `evaluation` split = test set, used once per final model | Tune and pick checkpoints on `validation` only. |
| Macro-F1 next to accuracy | Classes are imbalanced (Rice ~280 vs Dessert/Soup ~1,500). |
| Imbalance handled in training (weighted loss or sampler) | Not by copying files on disk. |
| Augmentation and normalisation on the fly | Not baked into `data/processed`. |

## How it works

- **Data:** `download_data.py` → `data/raw/food11` (9,866 / 3,430 / 3,347).
  `prepare_data.py` → `data/processed/food11`: removes ~630 pixel-identical duplicates (564 test
  images also appeared in training/validation; a duplicate stays in the earliest split), resizes
  the shorter side to 256 px, renames classes to snake_case. Result: 9,856 / 3,374 / 2,783.
- **Normalisation:** `FOOD11_MEAN/STD` in `src/data/constants.py` (recompute with
  `compute_stats.py` after rebuilding the data); the pretrained model uses ImageNet statistics.
- **Training:** experiments are YAML files in `configs/` loaded into `TrainConfig` (unknown keys
  are rejected). `scripts/train.py` checks data and GPU, runs `fit()` from `src/training/trainer.py`,
  then evaluates `best.pt` on the validation split. Validation runs every epoch; bfloat16 AMP on
  GPUs that support it; persistent DataLoader workers. Scratch: SGD + OneCycle, 80 epochs, early
  stopping off (`early_stopping_patience: null`) so the low-LR phase is reached. Pretrained:
  AdamW + cosine, 25 epochs, patience 7.
- **Tuning:** `scripts/tune.py` runs the trials of a sweep file (`configs/tuning/*.yaml`: base
  config + shared overrides such as a shorter `epochs` + per-trial changes) through `fit()`,
  into `checkpoints/tuning/` and `runs/tuning/`. Finished trials are reused, interrupted ones
  resumed. Selection is by validation macro-F1 only; `best_config.yaml` keeps the base epochs.
  Tunable augmentation: `randaugment_magnitude`, `crop_min_scale`, `random_erasing_p`,
  `mixup_alpha`, `cutmix_alpha` (MixUp/CutMix mix batches in `train_one_epoch`).
- **Imbalance:** the same `imbalance` value goes to `create_dataloaders` and `build_criterion`;
  default `weighted_loss`. Never combine loss weights and the sampler.
- **Run outputs** (git-ignored): `checkpoints/<experiment>/` — `best.pt` (weights + metadata:
  image size, normalisation, epoch, val metrics), `last.pt` (for `--resume`), `history.json`,
  `summary.json`; `runs/<experiment>/` TensorBoard; `reports/` evaluation and comparison output.
  A run is never overwritten; resuming requires the same config.
- **Using models:** all checkpoint loading goes through `src/inference/checkpoints.py`;
  preprocessing comes from the checkpoint metadata, never from extra arguments. CLI, API and
  demo require a trained checkpoint and never fall back to random or ImageNet-only weights.
  Uploads are decoded with `src/inference/images.py`.
- **Docker:** CPU images for API and demo (`Dockerfile`, `compose.yaml`); the checkpoint is
  mounted read-only as `/models/best.pt`. Compose reads `MODEL_CHECKPOINT` and ports from `.env`.
- **GPU:** on Windows `uv` installs the CUDA 13.2 PyTorch build (`[tool.uv.sources]`).

## Plan and status

1. ✅ Setup, dataset download, processing, EDA notebook
2. ✅ Data pipeline and class-imbalance handling (#4, #5); loss vs sampler comparison not run yet
3. ✅ Model factory (#6), training loop (#7)
4. ✅ Train ResNet-18 from scratch (#8) — first run early-stopped at epoch 44/80 (val macro-F1
   0.720); full 80-epoch rerun pending, then fill the README results
5. ✅ Pretrained reference (#9) — val macro-F1 0.917, accuracy 0.915; comparison chart via
   `compare_runs.py`
6. ⬜ Final test-split evaluation and error analysis (confusion matrix, Grad-CAM) for both models
7. ✅ Inference CLI, FastAPI, Streamlit demo (#11, #13, #22), checked with trained checkpoints
8. ✅ ONNX export (#12) — not used by the API/demo yet
9. ✅ Docker images, Compose, smoke tests (#14)
10. ✅ Hyper-parameter tuning tooling (#10) — `scripts/tune.py` + `configs/tuning/scratch.yaml`;
    running the sweep (~1.5 h) is optional and has not been done

Tasks are tracked as GitHub issues; reference the issue in the PR.

## Repository layout

```
configs/        experiment configs (YAML); tuning/ holds sweep files
scripts/        every user-facing command: download_data, prepare_data, compute_stats, train,
                compare_runs, tune, evaluate, predict, export_onnx
src/
  data/         dataset, transforms, dataloaders (training + validation only), class balance
  models/       ResNet-18 factory
  training/     config, training loop, losses, optimisers/schedules, checkpoints
  evaluation/   metrics, report figures, Grad-CAM, run comparison
  inference/    checkpoint loading, predictor, upload decoding, ONNX export/runtime
app/api/        FastAPI service
app/ui/         Streamlit demo
docker/         container healthcheck and smoke test (Dockerfile, compose.yaml at the root)
notebooks/      EDA (one notebook with outputs)
tests/          unit/ (fast, synthetic) and integration/ (training, ONNX, Streamlit)
```

ML logic lives in `src/`; `scripts/` and `app/` only call it. Every command is a script in
`scripts/` run as `uv run scripts/<name>.py` — no `python -m src...` entry points.

## Conventions

- Python 3.13, **uv** (`pyproject.toml` + `uv.lock`); add packages with `uv add`, never edit
  `uv.lock` by hand.
- **ruff** (line length 100) and **pytest**; tests must not need the real dataset or a GPU.
- English in code, commit messages and docs. Small functions, type hints.
- **No comments or docstrings unless something is genuinely non-obvious.**
- User-facing docs: `README.md` (what the project is, results) and `USAGE.md` (commands only).

## Git workflow

- `main` stable, updated only from `dev`; `dev` is the integration branch; one branch and PR
  per task. PR descriptions: change, validation, related issue.
- For each task, commit validated changes, push the feature branch, and open a PR targeting `dev`.
  If committing, pushing, or opening the PR is not possible, tell the user which step failed and why.
- Never commit `.env`, tokens, `data/`, `checkpoints/`, `runs/`, `reports/`, weights.

## Notes for agents

- Do not change the decisions table on your own; propose it in the PR description.
- Do not re-download the dataset or delete `data/` unless asked (`--verify-only` checks it).
- Long-running commands (training) are started by a human; agents may run short smoke tests.
- Update **Plan and status** when a task is done.
