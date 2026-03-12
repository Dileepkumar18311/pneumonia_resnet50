# Pneumonia Detection from Chest X-Rays using ResNet50

This repository contains a deep learning pipeline for detecting pneumonia from chest X-ray images. It leverages a pre-trained **ResNet50** model, fine-tuned on the Kermany et al. Chest X-Ray dataset to achieve high-accuracy binary classification (Normal vs. Pneumonia).

## 📊 Dataset
The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.
* **Format:** JPEG images
* **Categories:** 2 (Normal, Pneumonia)
* **Structure:** The data is organized into three folders (`train`, `test`, `val`) and contains subfolders for each image category.

## 🛠️ Project Structure
```text
├── train.py                 # Script for data augmentation, model building, and training
├── evaluate.py              # Script for testing the model and generating metrics
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files (datasets, saved models, etc.)
└── chest_xray/              # The dataset directory (must be downloaded separately)
    ├── train/
    ├── val/
    └── test/