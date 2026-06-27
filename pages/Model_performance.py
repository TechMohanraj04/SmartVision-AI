import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")

st.markdown("""
Compare the performance of all trained image classification models
and the YOLOv8 object detection model.
""")

# =====================================================
# LOAD RESULTS
# =====================================================

csv_path = "Models/model_performance_results/model_results.csv"

if not os.path.exists(csv_path):
    st.error("model_results.csv not found")
    st.stop()

df = pd.read_csv(csv_path)

# Rename only if needed
if "F1" in df.columns:
    df.rename(columns={"F1": "F1 Score"}, inplace=True)

if "Inference Time" in df.columns:
    df.rename(
        columns={
            "Inference Time": "Inference Time (s)"
        },
        inplace=True
    )

# =====================================================
# METRICS TABLE
# =====================================================

st.header("📋 Classification Model Metrics")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# SUMMARY CARDS
# =====================================================

st.header("📌 Summary")

best_accuracy = df.loc[
    df["Accuracy"].idxmax(),
    "Model"
]

best_f1 = df.loc[
    df["F1 Score"].idxmax(),
    "Model"
]

fastest_model = df.loc[
    df["Inference Time (s)"].idxmin(),
    "Model"
]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Best Accuracy",
        best_accuracy
    )

with c2:
    st.metric(
        "Best F1 Score",
        best_f1
    )

with c3:
    st.metric(
        "Fastest Model",
        fastest_model
    )

# =====================================================
# ACCURACY
# =====================================================

st.header("📈 Accuracy Comparison")

fig1, ax1 = plt.subplots(figsize=(8,4))

ax1.bar(
    df["Model"],
    df["Accuracy"]
)

ax1.set_ylabel("Accuracy")

st.pyplot(fig1)

# =====================================================
# PRECISION
# =====================================================

st.header("🎯 Precision Comparison")

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.bar(
    df["Model"],
    df["Precision"]
)

ax2.set_ylabel("Precision")

st.pyplot(fig2)

# =====================================================
# RECALL
# =====================================================

st.header("📌 Recall Comparison")

fig3, ax3 = plt.subplots(figsize=(8,4))

ax3.bar(
    df["Model"],
    df["Recall"]
)

ax3.set_ylabel("Recall")

st.pyplot(fig3)

# =====================================================
# F1 SCORE
# =====================================================

st.header("🏆 F1 Score Comparison")

fig4, ax4 = plt.subplots(figsize=(8,4))

ax4.bar(
    df["Model"],
    df["F1 Score"]
)

ax4.set_ylabel("F1 Score")

st.pyplot(fig4)

# =====================================================
# INFERENCE TIME
# =====================================================

st.header("⚡ Inference Speed")

fig5, ax5 = plt.subplots(figsize=(8,4))

ax5.bar(
    df["Model"],
    df["Inference Time (s)"]
)

ax5.set_ylabel("Seconds")

st.pyplot(fig5)

# =====================================================
# YOLO SECTION
# =====================================================

st.markdown("---")

st.header("🎯 YOLOv8 Object Detection Performance")

yolo_metrics = pd.DataFrame({

    "Metric": [
        "Precision",
        "Recall",
        "mAP@0.5",
        "mAP@0.5:0.95"
    ],

    "Value": [
        0.274,
        0.415,
        0.335,
        0.148
    ]
})

st.table(yolo_metrics)

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.markdown("---")

st.header("🔍 Confusion Matrix")

cm_path = "Models/model_performance_results/cm_dict.pkl"

if os.path.exists(cm_path):

    with open(cm_path, "rb") as f:
        cm_dict = pickle.load(f)

    model_name = st.selectbox(
        "Select Model",
        list(cm_dict.keys())
    )

    fig_cm, ax_cm = plt.subplots(
        figsize=(10,8)
    )

    sns.heatmap(
        cm_dict[model_name],
        cmap="Blues",
        ax=ax_cm
    )

    ax_cm.set_title(
        f"{model_name} Confusion Matrix"
    )

    st.pyplot(fig_cm)

else:
    st.warning(
        "cm_dict.pkl not found"
    )

# =====================================================
# BEST MODEL
# =====================================================

st.markdown("---")

best_model = df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Model"]

st.success(
    f"🏆 Best Classification Model: {best_model}"
)

st.info(
    "YOLOv8 was used for object detection tasks."
)