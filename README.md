# 🇮🇳 Indian Currency Classification using CNN (ResNet-18)

## 1. Introduction

Indian currency recognition is a practical computer vision problem with real-world applications such as:
- Automated cash handling systems
- Assistive technology for visually impaired users
- Banking and financial automation
- Fraud detection and verification systems

This project implements an **end-to-end deep learning pipeline** to classify **Indian currency denominations** from images using a **Convolutional Neural Network (CNN)** based on **ResNet-18** with transfer learning.

The project is designed with **clean architecture, proper evaluation, and generalization in mind**, making it suitable for learning, portfolios, and interviews.

---

## 2. Problem Statement

Given an image of an Indian currency note, the system should:
- Correctly identify the **denomination of the currency**
- Generalize well to unseen images
- Avoid overfitting
- Provide reliable evaluation on validation and test data

---

## 3. Supported Currency Denominations

The model classifies the following **7 Indian currency denominations**:

- ₹10  
- ₹20  
- ₹50  
- ₹100  
- ₹200  
- ₹500  
- ₹2000  

---

## 4. Dataset Description

- The dataset consists of labeled images of Indian currency notes.
- Images are divided into **training**, **validation**, and **test** sets.
- CSV files store image paths and labels.
- Raw images are intentionally **not included in the repository** due to size constraints.

### Dataset Splits
- Train set: Used to train the model
- Validation set: Used to monitor overfitting and tune performance
- Test set: Used for final evaluation

> Only CSV split files are committed to GitHub to keep the repository lightweight.

---

## 5. Label Encoding Strategy

Currency labels are encoded internally to work with PyTorch’s loss functions.

| Denomination | Encoded Label |
|-------------|---------------|
| ₹10  | 0 |
| ₹20  | 1 |
| ₹50  | 2 |
| ₹100 | 3 |
| ₹200 | 4 |
| ₹500 | 5 |
| ₹2000 | 6 |

This encoding is handled inside the Dataset class, keeping CSV files human-readable.

---

## 6. Model Architecture

### Base Model
- **ResNet-18**
- Pretrained on **ImageNet**
- Uses residual connections to avoid vanishing gradients

### Transfer Learning
- Pretrained weights are reused
- Final fully connected layer is replaced with a custom classifier

### Custom Classification Head
- Dropout (p = 0.5)
- Fully connected layer with 7 outputs

### Why ResNet-18?
- Lightweight and fast
- Strong performance on small to medium datasets
- Less prone to overfitting than deeper variants
- Widely used in industry and research

---

## 7. Training Configuration

- Epochs: **15**
- Batch size: **32**
- Optimizer: **Adam**
- Learning rate: **0.00005**
- Loss function: **CrossEntropyLoss**
- Regularization:
  - Dropout
  - Weight decay (L2 regularization)

Training logs include:
- Epoch number
- Training loss and accuracy
- Validation loss and accuracy

---

## 8. Overfitting Prevention

Several strategies were used to prevent overfitting:
- Transfer learning instead of training from scratch
- Dropout to prevent neuron co-adaptation
- Weight decay to control large weights
- Validation-based monitoring
- Early stopping readiness (optional extension)

The final model shows minimal gap between training and validation accuracy.

---

## 9. Model Evaluation

The trained model is evaluated on a **held-out test dataset** to measure real generalization.

Evaluation metrics include:
- Overall test accuracy
- Precision per class
- Recall per class
- F1-score per class
- Macro and weighted averages

This ensures balanced performance across all denominations.

---

## 10. Inference Pipeline

The inference pipeline:
1. Loads the trained model
2. Applies the same preprocessing used during training
3. Runs a forward pass
4. Returns the predicted denomination

Example usage:
```python
from src.inference.predict import load_model, predict_currency

model = load_model("best_currency_model.pth")
result = predict_currency("sample_image.jpg", model)
print(result)
```
---
## 11. Project Structure

currency-classification/
│
├── data/
│   └── processed/
│       └── splits/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
│
├── src/
│   ├── training/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── utils/
│       ├── logger.py
│       └── exception.py
│
├── notebook/
├── requirements.txt
├── README.md
└── .gitignore

---


## 12. How to Run the Project

### Step 1: Create Virtual Environment
```bash
python -m venv projectenv
projectenv\Scripts\activate

# Step 2: Install Dependencies
pip install -r requirements.txt

# Step 3: Train the Model
python -m src.training.train

# Step 4: Evaluate on Test Data
python -m src.training.evaluate
```
```markdown
## 13. Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Pandas  
- Scikit-learn  
- Pillow (PIL)  
- Deep Learning (CNNs)
## 14. Results Summary

- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Test Accuracy: ~99%
- Strong generalization
- Clean and modular codebase

## 15. Limitations

- Model is trained only on Indian currency
- Performance depends on image quality
- Extreme lighting or occlusion may affect predictions

### 16. Future Enhancements

- Unknown / foreign currency detection
- Real-time camera-based inference
- Web interface using Streamlit
- REST API using FastAPI
- Multi-country currency classification
- Deployment on cloud or edge devices

## 17. Author
Bhaskar Phaneendra T
Aspiring Data Scientist / Machine Learning Engineer

## 18. License
This project is licensed under the MIT License.