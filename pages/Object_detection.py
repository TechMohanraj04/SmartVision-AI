import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLOv8 Object Detection")

st.markdown("""
Upload an image and detect multiple objects using the trained YOLOv8 model.
""")

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = r"Models/runs/detect/models/yolo_runs/smartvision_yolo-4/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found:\n{MODEL_PATH}")
    st.stop()

model = YOLO(MODEL_PATH)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.50,
    step=0.05
)

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# DETECTION
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # YOLO Prediction

    results = model.predict(
        source=np.array(image),
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    st.write("Boxes object:", result.boxes)
    st.write("Number of boxes:", len(result.boxes))

    annotated_img = result.plot()

    with col2:
        st.subheader("🎯 Detection Result")

        st.image(
            annotated_img,
            use_container_width=True
        )

    # =================================================
    # DETECTION DETAILS
    # =================================================

    boxes = result.boxes

    st.markdown("---")

    st.subheader("📦 Detection Summary")

    if len(boxes) > 0:

        detected_objects = []

        for box in boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            confidence = float(box.conf[0]) * 100

            detected_objects.append({
                "Object": class_name,
                "Confidence (%)": round(confidence, 2)
            })

        df = pd.DataFrame(detected_objects)

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Total Objects Detected",
                len(df)
            )

        with c2:
            st.metric(
                "Unique Classes",
                df["Object"].nunique()
            )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"Successfully detected {len(df)} object(s)"
        )

    else:

        st.warning(
            "No objects detected in the image."
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.info(
    "YOLOv8 • Smart Vision AI Object Detection Module"
)