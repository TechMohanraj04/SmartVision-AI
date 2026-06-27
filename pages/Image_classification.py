import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
import torch.nn as nn
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Image Classification",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Multi-Model Image Classification")

st.markdown("""
Upload an image and compare predictions from multiple
deep learning models.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🤖 Smart Vision AI")

st.sidebar.info("""
Supported Models:

✅ VGG16

✅ ResNet50

✅ MobileNetV2

✅ EfficientNetB0
""")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.sidebar.success(f"Device: {device}")

# =====================================================
# CLASS NAMES
# =====================================================

TRAIN_PATH = r"smartvision_dataset/classification/train"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

CLASS_NAMES = train_dataset.classes
num_classes = len(CLASS_NAMES)

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# LOAD VGG16
# =====================================================


@st.cache_resource
def load_vgg16():

    model = models.vgg16(weights=None)

    # Architecture reconstructed from checkpoint
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    )

    checkpoint = torch.load(
        r"Models/models/vgg16_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# =====================================================
# LOAD RESNET50
# =====================================================

@st.cache_resource
def load_resnet50():

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    checkpoint = torch.load(
        r"Models/models/resnet50_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# =====================================================
# LOAD MOBILENETV2
# =====================================================

@st.cache_resource
def load_mobilenet():

    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    checkpoint = torch.load(
        r"Models/models/mobilenetv2_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# =====================================================
# LOAD EFFICIENTNETB0
# =====================================================

@st.cache_resource
def load_efficientnet():

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    checkpoint = torch.load(
        r"Models/models/efficientnetb0_best.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# =====================================================
# LOAD ALL MODELS
# =====================================================

with st.spinner("Loading Models..."):

    vgg16 = load_vgg16()
    resnet50 = load_resnet50()
    mobilenetv2 = load_mobilenet()
    efficientnetb0 = load_efficientnet()

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict(model, image_tensor):

    with torch.no_grad():

        outputs = model(image_tensor)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        top5_probs, top5_indices = torch.topk(
            probs,
            5
        )

    predictions = []

    for prob, idx in zip(
        top5_probs[0],
        top5_indices[0]
    ):

        predictions.append({

            "class":
            CLASS_NAMES[idx.item()],

            "confidence":
            float(prob) * 100
        })

    return predictions

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# INFERENCE
# =====================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns([1,2])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    predictions = {

        "VGG16":
            predict(vgg16, image_tensor),

        "ResNet50":
            predict(resnet50, image_tensor),

        "MobileNetV2":
            predict(mobilenetv2, image_tensor),

        "EfficientNetB0":
            predict(efficientnetb0, image_tensor)
    }

    st.markdown("---")

    st.subheader("📊 Model Predictions")

    cols = st.columns(4)

    comparison_data = []

    for idx, (
        model_name,
        preds
    ) in enumerate(
        predictions.items()
    ):

        with cols[idx]:

            st.markdown(
                f"### {model_name}"
            )

            st.success(
                preds[0]["class"]
            )

            st.metric(
                "Confidence",
                f"{preds[0]['confidence']:.2f}%"
            )

            st.markdown(
                "#### Top-5 Predictions"
            )

            for pred in preds:

                st.write(
                    f"{pred['class']} "
                    f"({pred['confidence']:.2f}%)"
                )

        comparison_data.append({

            "Model":
            model_name,

            "Prediction":
            preds[0]["class"],

            "Confidence (%)":
            round(
                preds[0]["confidence"],
                2
            )
        })

    st.markdown("---")

    st.subheader(
        "🏆 Model Comparison"
    )

    df = pd.DataFrame(
        comparison_data
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    best_model = max(
        comparison_data,
        key=lambda x:
        x["Confidence (%)"]
    )

    st.success(
        f"Best Prediction: "
        f"{best_model['Prediction']} "
        f"by {best_model['Model']} "
        f"({best_model['Confidence (%)']:.2f}%)"
    )