# Usage

All commands run from the repository root.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and, for training, an NVIDIA GPU.

```bash
uv sync --all-extras
cp .env.example .env
```

Put your Kaggle token (https://www.kaggle.com/settings/api) into `KAGGLE_API_TOKEN` in `.env`.

Check the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Data

```bash
uv run scripts/download_data.py
uv run scripts/prepare_data.py
```

Options: `download_data.py --force | --verify-only`, `prepare_data.py --size 256`.
After changing the preparation, update the normalisation values in `src/data/constants.py`:

```bash
uv run scripts/compute_stats.py
```

## Training

```bash
uv run scripts/train.py
uv run scripts/train.py --config configs/resnet18_pretrained.yaml
```

Continue an interrupted run:

```bash
uv run scripts/train.py --resume
uv run scripts/train.py --config configs/resnet18_pretrained.yaml --resume
```

Training curves:

```bash
uv run tensorboard --logdir runs
```

Compare both runs (writes `reports/comparison/val_curves.png` and `results.md`):

```bash
uv run scripts/compare_runs.py
```

To retrain an experiment from scratch, delete `checkpoints/<experiment>`, `runs/<experiment>`
and `reports/<experiment>` first.

## Evaluation

Validation split:

```bash
uv run scripts/evaluate.py --checkpoint checkpoints/resnet18_scratch/best.pt --output-dir reports/resnet18_scratch/val
```

Test split — once per final model:

```bash
uv run scripts/evaluate.py --checkpoint checkpoints/resnet18_scratch/best.pt --split evaluation --output-dir reports/resnet18_scratch/test
```

## Prediction

```bash
uv run scripts/predict.py --image path/to/dish.jpg
uv run scripts/predict.py --dir path/to/photos --model checkpoints/resnet18_pretrained/best.pt
```

Options: `--top-k N`, `--json`, `--device cpu|cuda`. `--model` also accepts an exported `model.onnx`.

## API

```bash
uv run uvicorn app.api.main:app
```

Docs: http://localhost:8000/docs. Another model: set `CHECKPOINT_PATH` before starting.

## Demo

```bash
uv run streamlit run app/ui/streamlit_app.py
```

Opens at http://localhost:8501.

## ONNX export

```bash
uv run scripts/export_onnx.py --checkpoint checkpoints/resnet18_scratch/best.pt --output-dir checkpoints/onnx_scratch
uv run scripts/predict.py --image path/to/dish.jpg --model checkpoints/onnx_scratch/model.onnx
```

## Docker

Serves the API and demo with the model from `MODEL_CHECKPOINT` in `.env`
(default `checkpoints/resnet18_scratch/best.pt`).

```bash
docker compose up --build -d --wait
docker compose down
```

API: http://localhost:8000/docs · Demo: http://localhost:8501

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Docker images and container smoke test:

```bash
docker build --target api -t food-classifier-api:local .
docker build --target demo -t food-classifier-demo:local .
python docker/smoke_test.py
```
