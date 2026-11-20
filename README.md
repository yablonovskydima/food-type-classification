# Food Classifier

Classifies a food photo into one of the 11 [Food-11](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
categories: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood,
Soup, Vegetable/Fruit.

The main model is **ResNet-18 trained from scratch**. The same architecture fine-tuned from
**ImageNet weights** is trained as a reference to show what pretraining adds.

How to run everything: **[USAGE.md](USAGE.md)**.

## Approach

- **Data** — Food-11 contains ~630 duplicate images across splits (564 test photos also appear
  in training/validation). They are removed, so the test set does not leak. Images are resized
  to a 256 px shorter side. Final split: 9,856 train / 3,374 validation / 2,783 test.
- **Training** — PyTorch, own training loop: strong augmentation, class-weighted loss for the
  imbalanced classes, label smoothing, mixed precision, validation every epoch, best model by
  validation macro-F1. Settings live in `configs/*.yaml`.
- **Evaluation** — accuracy, top-3, macro-F1, per-class precision/recall/ROC-AUC, confusion
  matrix and Grad-CAM of the most confident mistakes. The test split is used once per final model.
- **Serving** — command-line prediction, FastAPI service, Streamlit demo, Docker images, ONNX export.

## Results

Validation split, best epoch by macro-F1 (RTX 3050).

| Model | Training time | Epochs (best) | Val accuracy | Val top-3 | Val macro-F1 |
|-------|---------------|---------------|--------------|-----------|--------------|
| ResNet-18 from scratch | — | — | — | — | — |
| ResNet-18 ImageNet-pretrained | 10.2 min | 25 (22) | 0.915 | 0.981 | 0.917 |

The from-scratch row is filled in after its full 80-epoch run.

## Project structure

```
configs/      experiment configs
scripts/      all commands (data, training, evaluation, prediction, export)
src/          data, models, training, evaluation, inference
app/          FastAPI service (api/) and Streamlit demo (ui/)
docker/       container healthcheck and smoke test
notebooks/    exploratory data analysis
tests/        unit and integration tests
```

Generated folders (git-ignored): `data/`, `checkpoints/`, `runs/`, `reports/`.

## Tech stack

Python 3.13 · uv · PyTorch / torchvision · scikit-learn · TensorBoard · FastAPI · Streamlit ·
ONNX Runtime · Docker

## License

Code: MIT. Dataset: Food-11, MMSPG, EPFL — Singla A., Yuan L., Ebrahimi T., *Food/Non-food Image
Classification and Food Categorization using Pre-Trained GoogLeNet Model*, MADiMa 2016.
