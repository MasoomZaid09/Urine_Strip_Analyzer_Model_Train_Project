# Urine Analyse Model Project 🧪

This project trains a **deep learning-based image classification model** to analyze **urine test strips**. The model is trained on images from RoboFlow and exported to **TensorFlow Lite (TFLite)** for deployment in mobile/embedded applications.

---

## 📂 Project Structure

When you add the dataset from RoboFlow, your project should look like this:

```
Urine_Analyse_Model_Project/
│── train_model.py        # Main training script
│── _annotations.csv      # Dataset annotations (from RoboFlow)
│── train/                # Training images (from RoboFlow)
```

After running the training script, the following files will be generated:

```
│── labels.txt            # Class labels
│── model.tflite          # TFLite model for deployment
│── saved_model.keras     # Trained Keras model
```

---

## 🛠️ Installation & Setup

```bash
# Clone this repository
git clone https://github.com/<your-username>/Urine_Analyse_Model_Project.git
cd Urine_Analyse_Model_Project

# Create virtual environment
python3 -m venv tflite_env
source tflite_env/bin/activate

# Install dependencies
pip install tensorflow-macos==2.13.0 keras==2.13.1 pandas numpy scikit-learn matplotlib
```

---

## 📊 Dataset

The dataset is available on **[RoboFlow Universe](https://universe.roboflow.com/urine-test-strips-qq9jx/urine-test-strips-main-2)**.  

Steps:
1. Download the dataset from RoboFlow.  
2. Place `_annotations.csv` and the `train/` folder into your project directory.  

---

## 🏗️ Model Architecture

This project uses a **Convolutional Neural Network (CNN)**:

- Input Layer → RGB image resized to **224x224**  
- Conv2D (32 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Conv2D (64 filters, 3×3) + ReLU  
- MaxPooling2D (2×2)  
- Flatten  
- Dense (128 units, ReLU)  
- Dense (N classes, Softmax) → Output layer  

**Training Setup**:  
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metric: Accuracy  
- Epochs: 10  

---

## ▶️ Training the Model

Run:

```bash
python3 train_model.py
```

The script will:  
- Load images + labels from `_annotations.csv` and `train/`.  
- Save class labels into `labels.txt`.  
- Save trained model as `saved_model.keras`.  
- Export to `model.tflite` for deployment.  

---

## 📦 Outputs

- `labels.txt` → Class labels  
- `saved_model.keras` → Full Keras model  
- `model.tflite` → Optimized TFLite model  

---

## 📌 Reference

Dataset Source 👉 [RoboFlow Universe - Urine Test Strips](https://universe.roboflow.com/urine-test-strips-qq9jx/urine-test-strips-main-2)  

---

## 💡 Future Work

- Improve accuracy with transfer learning (MobileNet/EfficientNet).  
- Add data augmentation for better generalization.  
- Deploy in a mobile app for real-time strip analysis.  

---

## 👨‍💻 Author

**Masoom Zaid**  
Passionate Android & Cross-Platform Developer | Exploring AI in Healthcare 🚀
