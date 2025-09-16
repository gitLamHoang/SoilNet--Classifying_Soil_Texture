# SoilNet – Soil Texture Classification using CNNs

SoilNet is a deep learning project for classifying soil textures into three categories — **Coarse**, **Medium**, and **Fine** — using Convolutional Neural Networks (CNNs). The model achieves **99.46% accuracy** on the SOIL (v1) dataset, demonstrating CNNs' potential for fast, consistent, and automated soil texture classification.

---

## 📌 Background & Motivation
Soil texture significantly affects water retention, nutrient dynamics, and agricultural productivity. Traditional classification methods are time-consuming, require expert judgment, and can be subjective. This project leverages CNNs to automate the classification process, delivering rapid and highly accurate predictions.

---

## 📊 Dataset
We used a combination of:
- **SOIL (v1) Dataset** (Roboflow, 2025) – 3,573 soil images  
- **Kaggle Soil Image Dataset** (Pondy, 2022) – additional samples to improve diversity  

### Preprocessing Steps:
- Removed irrelevant labels (e.g., `sandy soil`)
- Relabeled images into **coarse**, **medium**, and **fine**
- Resized all images to **256×256**
- One-hot encoded class labels
- Split data into **80% training** / **20% testing**

---

## 🏗️ Model Architecture
The CNN consists of **four convolutional blocks** for hierarchical feature extraction, followed by a fully connected classification head.

- **Conv Block 1:** 16 filters, 3×3 kernel, ReLU + BatchNorm + MaxPool  
- **Conv Block 2:** 32 filters, 3×3 kernel, ReLU + BatchNorm + MaxPool  
- **Conv Block 3:** 64 filters, 3×3 kernel, ReLU + BatchNorm + MaxPool  
- **Conv Block 4:** 128 filters, 3×3 kernel, ReLU + BatchNorm + Adaptive Avg Pool  
- **Fully Connected Layer:** 128 → 3 (Softmax)

---

## ⚙️ Hyperparameters
| Parameter           | Values Tested        | Best Value |
|--------------------|-------------------|-----------|
| Epochs            | 30, 50, 70        | **50** |
| Learning Rate     | 0.01, 0.001, 0.0001 | **0.001** |
| Kernel Size       | 3×3, 5×5          | **5×5** |
| Activation        | ReLU, LeakyReLU   | **LeakyReLU** |

---

## 📈 Results
| Metric        | Score |
|--------------|------|
| Accuracy     | **99.46%** |
| Precision    | **0.9946** |
| Recall       | **0.9946** |
| F1-Score     | **0.9946** |
| Test Loss    | **0.0182** |
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/27ef6182-f7c9-4af6-807b-f65f1aa489b8" />


### Confusion Matrix
- **Coarse → Medium:** 1 misclassification  
- **Medium → Coarse:** 3 misclassifications  
- **Fine:** 100% correctly classified  
<img width="553" height="455" alt="image" src="https://github.com/user-attachments/assets/753e069e-2cb7-4604-b33c-a7e75bbd88c3" />

---

## 🚀 Setup & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/SoilNet.git
cd SoilNet
