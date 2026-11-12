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

## Roadmap

- [ ] Data pipeline and EDA
- [ ] ResNet-18 trained from scratch
- [ ] Scratch vs pretrained comparison
- [ ] Error analysis, confusion matrix, Grad-CAM
- [ ] ONNX export, inference API, Gradio demo

---

## License & credits

- Code: MIT
- Dataset: Food-11, Multimedia Signal Processing Group (MMSPG), EPFL.
  Singla A., Yuan L., Ebrahimi T. *Food/Non-food Image Classification and Food Categorization using Pre-Trained GoogLeNet Model*, MADiMa 2016.
  Use subject to the dataset's terms.
