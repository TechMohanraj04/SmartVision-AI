import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="About Smart Vision AI",
    page_icon="📘",
    layout="wide"
)

st.title("📘 About Smart Vision AI")

st.markdown("---")

# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.header("🧠 Project Overview")

st.markdown("""
**Smart Vision AI** is an end-to-end Computer Vision application that combines
Image Classification and Object Detection using state-of-the-art Deep Learning models.

The system enables users to:

- Classify images into predefined object categories
- Detect and localize multiple objects in images
- Compare predictions across multiple CNN architectures
- Visualize model performance through interactive dashboards

This project demonstrates the practical application of
Deep Learning, Transfer Learning, and Computer Vision techniques.
""")

# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("📂 Dataset Information")

st.markdown("""
### Dataset Source
Custom subset derived from the COCO Dataset.

### Number of Classes
26 Classes

### Categories Included

- airplane
- bed
- bench
- bicycle
- bird
- bottle
- bowl
- bus
- cake
- car
- cat
- chair
- couch
- cow
- cup
- dog
- elephant
- horse
- motorcycle
- person
- pizza
- potted plant
- stop sign
- traffic light
- train
- truck

### Tasks Performed

✅ Image Classification

✅ Object Detection
""")

# =====================================================
# MODEL ARCHITECTURES
# =====================================================

st.header("🤖 Deep Learning Models")

st.markdown("""
### Image Classification Models

1. VGG16
2. ResNet50
3. MobileNetV2
4. EfficientNetB0

### Object Detection Model

- YOLOv8

### Techniques Used

- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Early Stopping
- Learning Rate Scheduling
- Model Evaluation & Comparison
""")

# =====================================================
# PERFORMANCE SUMMARY
# =====================================================

st.header("📊 Performance Summary")

st.markdown("""
### Classification Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Inference Time

### Object Detection Evaluation Metrics

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

### Key Highlights

- Multi-model comparison dashboard
- Real-time image classification
- YOLOv8 object detection
- Interactive Streamlit deployment
""")

# =====================================================
# TECH STACK
# =====================================================

st.header("🛠️ Technology Stack")

st.markdown("""
### Programming Language

- Python

### Deep Learning Frameworks

- PyTorch
- Torchvision
- Ultralytics YOLO

### Libraries

- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow

### Deployment

- Streamlit

### Development Environment

- Google Colab
- VS Code
""")

# =====================================================
# PROJECT FEATURES
# =====================================================

st.header("🚀 Key Features")

st.markdown("""
- Multi-model Image Classification
- YOLOv8 Object Detection
- Interactive Streamlit Dashboard
- Model Performance Comparison
- Confusion Matrix Visualization
- Real-Time Inference Pipeline
- End-to-End Computer Vision Workflow
""")

# =====================================================
# DEVELOPER INFORMATION
# =====================================================

st.header("👨‍💻 Developer Information")

st.markdown("""
### Mohanraj D

**Project Name:** Smart Vision AI

**Domain:** Computer Vision & Deep Learning

**Educational Background:** B.Sc Computer Science

### Tools Used

- PyTorch
- YOLOv8
- Streamlit
- OpenCV
- VS Code
- Google Colab
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success(
    "Smart Vision AI • Computer Vision & Deep Learning Project"
)