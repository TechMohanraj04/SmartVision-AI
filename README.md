# 🚀 SmartVision AI

An AI-powered Computer Vision application built using **PyTorch, YOLOv8, Streamlit, and OpenCV** for:

* 🖼️ Image Classification
* 🎯 Object Detection
* 📊 Model Performance Comparison
* ⚡ Real-time Inference

The project compares multiple deep learning CNN architectures and integrates YOLOv8 for object detection in a complete Streamlit web application.

---

## 📌 Features

### ✅ Image Classification

* Upload images for prediction
* Compare predictions from:

  * VGG16
  * ResNet50
  * MobileNetV2
  * EfficientNetB0
* Top-5 predictions with confidence scores
* Side-by-side model comparison

### ✅ Object Detection

* YOLOv8 object detection
* Bounding boxes with labels
* Confidence score visualization
* Adjustable confidence threshold
* Non-Maximum Suppression (NMS)

### ✅ Model Performance Dashboard

* Accuracy comparison
* Precision, Recall, and F1-Score analysis
* Inference speed comparison
* Confusion matrices
* Model size analysis

### ✅ Streamlit Multi-Page Application

* 🏠 Home Page
* 🖼️ Image Classification
* 🎯 Object Detection
* 📊 Model Performance
* ℹ️ About Section

---

## 🧠 Models Used

| Model          | Task                 |
| -------------- | -------------------- |
| VGG16          | Image Classification |
| ResNet50       | Image Classification |
| MobileNetV2    | Image Classification |
| EfficientNetB0 | Image Classification |
| YOLOv8         | Object Detection     |

---

## 📂 Dataset Information

The project uses a custom-organized dataset for both image classification and object detection tasks.

### Classification Dataset Structure

```text
smartvision_dataset/
│
├── classification/
│   ├── train/
│   ├── val/
│   └── test/
```

### Detection Dataset Structure

```text
smartvision_dataset/
│
├── detection/
│   ├── images/
│   └── labels/
│       └── data.yaml
```

### Dataset Details

* Total Classes: **26**
* Classification Format: **PyTorch ImageFolder**
* Detection Format: **YOLOv8 Format**

### Dataset Download

The dataset is hosted on Google Drive due to its large size and is not included in this repository.

🔗 **Google Drive Dataset:**

https://drive.google.com/drive/folders/1BNgozSuiLISDMLDAAzbQJxcFCGTSfz2U?usp=drive_link

---

## 📊 Model Performance

| Model          | Strength                         |
| -------------- | -------------------------------- |
| VGG16          | Stable baseline model            |
| ResNet50       | Strong feature extraction        |
| MobileNetV2    | Lightweight and fast inference   |
| EfficientNetB0 | Balanced efficiency and accuracy |
| YOLOv8         | Real-time object detection       |

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchvision
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## ▶️ Run the Streamlit Application

```bash
streamlit run Home.py
```

---

## 📁 Project Structure

```text
SmartVision-AI/
│
├── pages/
│   ├── Image_classification.py
│   ├── Object_detection.py
│   ├── Model_performance.py
│   └── About.py
│
├── models/
│   ├── vgg16_best.pth
│   ├── resnet50_best.pth
│   ├── mobilenetv2_best.pth
│   ├── efficientnetb0_best.pth
│   └── yolo_runs/
│       ├── YOLOv8n-detection/
│       ├── YOLOv8s-detection/
│       └── YOLOv8m-detection/
│
├── smartvision_dataset/
├── Home.py
├── requirements.txt
└── README.md
```

---

## ⚡ Performance Optimization

* TorchScript model export
* Model quantization
* GPU acceleration
* Batch inference support
* Optimized memory usage

---

## 🤗 Live Demo

Hugging Face Space:

https://huggingface.co/spaces/Mohanrajdeena/Smartvision-AI

---

## 👨‍💻 Developer

**Mohanraj D**

🎓 B.Sc. Computer Science
📊 Data Analyst & AI Enthusiast
🤖 Machine Learning & Computer Vision Projects

GitHub: https://github.com/Mohanrajdeena

---

## 📌 Future Improvements

* Live webcam detection
* Video object detection
* Cloud deployment
* Model explainability (Grad-CAM)
* REST API integration
* Mobile application support

---

## 📜 License

This project is licensed under the MIT License and is intended for educational, research, and portfolio purposes.
