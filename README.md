
# 🩺 PneumoDetect - Pneumonia Detection from Chest X-rays using VGG16

A deep learning project to classify chest X-ray images as **Normal** or **Pneumonia** using **transfer learning** with the VGG16 model.

## 📁 Dataset
- Source: Kaggle – Chest X-ray Images (Pneumonia)
- Classes: NORMAL, PNEUMONIA
- Preprocessing: Rescaling, Data Augmentation (zoom, flip, brightness, rotation)
- Image Size: Resized to 224×224

## 🌟 Key Features
- Transfer learning using **VGG16** (pretrained on ImageNet)
- Custom classification head with **Dropout** for regularization
- **Data augmentation** to prevent overfitting
- **Confusion Matrix**, **ROC Curve**, and **Classification Report** for evaluation
- Predicts both **batch test sets** and **individual X-ray images**
- GUI built with **PyQt5** for real-time diagnosis
- **Text-to-Speech** feedback for accessibility
- **Grad-CAM visualization** for highlighting pneumonia regions
- Generates a detailed **PDF report** with diagnosis and doctor's recommendations
- Accuracy: **90.06%**

## 🧠 Model Summary
- Base: VGG16 (frozen)
- Top Layers: Flatten → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(2, softmax)
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical Crossentropy
- Epochs: 15 | Batch Size: 16

## 📊 Evaluation Metrics
- Accuracy: 90.06%
- F1-Score: NORMAL = 0.85, PNEUMONIA = 0.93
- Visual metrics and reports generated with matplotlib, seaborn, sklearn

## 🖥 GUI Application
- Upload X-ray image and get prediction
- Shows live Grad-CAM highlighting infection areas
- Generates and saves patient-specific **PDF report**
- Directs to appointment booking site if pneumonia is detected

## 🔮 Future Enhancements
- Try other architectures (e.g., ResNet, EfficientNet)
- Expand dataset and improve balance
- Deploy model as a full-stack web application

