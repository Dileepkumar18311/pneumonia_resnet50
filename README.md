# 🩺 Pneumonia Detection from Chest X-Rays using ResNet50

This project is a **Deep Learning–based medical imaging system** that detects **Pneumonia from Chest X-ray images**.  
It uses **Transfer Learning with ResNet50** and provides a **Streamlit web interface** for real-time predictions.

The goal of this project is to assist in **early pneumonia detection**, which can support medical professionals in diagnosis.

---

# 📊 Dataset Overview

The model is trained using the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

🔗 Dataset Link  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Dataset Details

- **Type:** Binary Classification  
- **Classes:**  
  - Normal  
  - Pneumonia
- **Total Images:** 5,856
- **Image Type:** Chest X-ray (JPEG)

### Dataset Structure

```

chest_xray/
│
├── train/
│   ├── NORMAL
│   └── PNEUMONIA
│
├── val/
│   ├── NORMAL
│   └── PNEUMONIA
│
└── test/
├── NORMAL
└── PNEUMONIA

```

---

# 🛠️ Tech Stack

### Deep Learning Framework
- TensorFlow
- Keras

### Model Architecture
- ResNet50 (Transfer Learning)

### Frontend
- Streamlit

### Programming Language
- Python

### Libraries Used
- NumPy
- Scikit-learn
- Pillow
- Matplotlib
- TensorFlow / Keras

### Model Storage
- Git LFS (Large File Storage)

---

# 🧠 Model Architecture

This project uses **Transfer Learning** with a **pretrained ResNet50 model**.

### Key Steps

1. Load **ResNet50 pretrained on ImageNet**
2. Remove the original classification layer
3. Add a custom classification head
4. Train the model on chest X-ray images

### Custom Classification Head

```

ResNet50 Base Model
↓
Global Average Pooling
↓
Dense Layer (ReLU)
↓
Dropout
↓
Dense Layer (Sigmoid)

```

---

# ⚙️ Training Pipeline

Training was performed in **two phases**.

## Phase 1: Initial Training

- ResNet50 base layers were **frozen**
- Only the **classification head** was trained
- Optimizer: **Adam**
- Learning Rate: **1e-4**

Purpose:
- Train the classifier without modifying pretrained features

---

## Phase 2: Fine-Tuning

- The **last 20 layers of ResNet50** were unfrozen
- Training continued with a **lower learning rate**

Optimizer: Adam  
Learning Rate: **1e-5**

Purpose:
- Allow the network to learn **X-ray specific patterns**

---

# 🛡️ Regularization Techniques

To prevent overfitting, the following techniques were used:

### Early Stopping

Stops training when validation loss stops improving.

```

EarlyStopping(patience=5)

```

### Model Checkpoint

Saves the **best performing model weights**.

```

ModelCheckpoint(save_best_only=True)

````

---

# 📈 Performance Results

| Metric | Score |
|------|------|
| Test Accuracy | ~91% |
| Pneumonia Recall | ~93% |

### Why Recall Matters

In medical diagnosis:

- **High Recall** ensures fewer **false negatives**
- This means pneumonia cases are **less likely to be missed**

---

# 🚀 Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Dileepkumar18311/pneumonia_resnet50.git
cd pneumonia_resnet50
````

---

## 2️⃣ Install Git LFS

This repository uses **Git Large File Storage** for the trained model.

```bash
git lfs install
git lfs pull
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

After running, open the browser:

```
http://localhost:8501
```

Upload a **Chest X-ray image**, and the model will predict:

* **Normal**
* **Pneumonia**

---

# 📂 Project Structure

```
pneumonia_resnet50/
│
├── train.py
│   Model training and fine-tuning pipeline
│
├── evaluate.py
│   Model evaluation (confusion matrix, classification report)
│
├── app.py
│   Streamlit web application for predictions
│
├── requirements.txt
│   Python dependencies
│
├── pneumonia_resnet50_model.keras
│   Trained model file (stored with Git LFS)
│
└── README.md
```

---

# 📊 Example Prediction Workflow

1. Upload chest X-ray image
2. Image preprocessing
3. Model inference
4. Prediction displayed in UI

```
Input Image
   ↓
Preprocessing
   ↓
ResNet50 Model
   ↓
Prediction
   ↓
Normal / Pneumonia
```

---

# 🔬 Future Improvements

Possible improvements for this project:

* Add **Grad-CAM visualization** for model explainability
* Improve dataset balancing
* Deploy using **Docker**
* Create a **REST API using FastAPI**
* Deploy on **AWS / GCP**

---

# 👨‍💻 Author

**Dileep Kumar**

🎓 B.S. Computer Science
Sukkur IBA University

💻 AI Engineer | Machine Learning Developer

GitHub:
[https://github.com/Dileepkumar18311](https://github.com/Dileepkumar18311)

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository on GitHub
🍴 Fork it for your own experiments

---

# 📜 License

This project is licensed under the **MIT License**.

