# Dense-Convoltuional-Networks-for-Breast-Cancer-Classification

A hybrid deep learning model combining DenseNet and Compact Convolutional Transformers (CCT) for multiclass classification of breast cancer histopathological images. This project focuses on improving diagnostic accuracy using SMOTE-based balancing and Macenko stain normalization on the BreakHis dataset.

---

## 📌 Objective

To accurately classify breast cancer images into **eight subtypes** using a hybrid model that leverages:

* DenseNet for feature extraction
* Compact Convolutional Transformers (CCT) for contextual encoding

---

## 🧠 Key Features

* 🔬 Multiclass classification (8 categories)
* 🧪 Preprocessing: Macenko stain normalization
* 🗁 Data balancing using SMOTE
* 🏗️ DenseNet + CCT hybrid model
* 📊 Evaluation with metrics like Accuracy, Precision, Recall, F1-score

---

## 📁 Repository Structure

| File/Folder        | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `main.py`          | Entry point of the training/inference pipeline             |
| `data_loader.py`   | Custom dataset loader with augmentations and preprocessing |
| `model.py`         | Definition of DenseNet-CCT hybrid architecture             |
| `train_eval.py`    | Training and evaluation loops                              |
| `utils.py`         | Helper functions (metrics, visualizations, etc.)           |
| `requirements.txt` | Required Python packages                                   |
| `README.md`        | Project overview and instructions                          |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Dense-Convoltuional-Networks-for-Breast-Cancer-Classification.git
cd Dense-Convoltuional-Networks-for-Breast-Cancer-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Dataset

Place the **BreakHis dataset** (or your preprocessed version) inside a folder named `data/` like:

```
data/
└── benign/
    └── subtype1/
    └── subtype2/
└── malignant/
    └── subtype3/
    └── ...
```

### 4. Train the Model

```bash
python main.py
```

---

## 📈 Evaluation Metrics

* Accuracy
* F1 Score (Macro & Weighted)
* Confusion Matrix
* ROC-AUC per class

---

## 📄 Citation

If you use this project for research, please cite or acknowledge the repository.

---

## ✍️ Author

Jacob Kevin
B.Tech CSE, SRM Institute of Science and Technology

---

## ⚠️ Disclaimer

This tool is for research purposes only and should not be used for real-world diagnosis without medical supervision.
