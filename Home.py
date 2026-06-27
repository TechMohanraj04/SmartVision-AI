import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Vision AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================

st.title("🧠 Smart Vision AI")
st.subheader("Image Classification & Object Detection System")

st.markdown("---")

# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.header("📌 Project Overview")

st.markdown("""
Smart Vision AI is a Computer Vision application that combines:

- 📸 Multi-Model Image Classification
- 🎯 YOLOv8 Object Detection
- 📊 Model Performance Analysis
- 🚀 Interactive Streamlit Dashboard

The project leverages Deep Learning and Transfer Learning techniques
to classify and detect objects from images.
""")

# =====================================================
# FEATURES
# =====================================================

st.header("🚀 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    ✅ Image Classification

    • VGG16

    • ResNet50

    • MobileNetV2

    • EfficientNetB0
    """)

with col2:
    st.success("""
    ✅ Object Detection

    • YOLOv8

    • Bounding Box Detection

    • Confidence Scores

    • Multiple Object Recognition
    """)

# =====================================================
# DATASET
# =====================================================

st.header("📂 Dataset Information")

st.info("""
Dataset: COCO-based Custom Dataset

Total Classes: 26

Tasks:
• Image Classification
• Object Detection
""")

# =====================================================
# AVAILABLE PAGES
# =====================================================

st.header("📑 Application Pages")

page1, page2 = st.columns(2)

with page1:
    st.info("""
📸 Image Classification

Upload an image and compare predictions from:

• VGG16
• ResNet50
• MobileNetV2
• EfficientNetB0
""")

with page2:
    st.info("""
🎯 Object Detection

Upload an image and detect objects using YOLOv8
with confidence scores and bounding boxes.
""")

page3, page4 = st.columns(2)

with page3:
    st.info("""
📊 Model Performance

Compare:

• Accuracy
• Precision
• Recall
• F1 Score
• Inference Time
""")

with page4:
    st.info("""
📘 About

Project details,
dataset information,
models used,
and developer information.
""")

# =====================================================
# HOW TO USE
# =====================================================

st.header("🛠 How To Use")

st.markdown("""
1. Select a page from the sidebar.
2. Upload an image.
3. Run inference.
4. View predictions and results.
5. Compare model performances.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🧭 Navigation")

st.sidebar.info("""
Select a page from the sidebar:

📸 Image Classification

🎯 Object Detection

📊 Model Performance

📘 About
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.success(
    "Smart Vision AI • Computer Vision & Deep Learning Project"
)