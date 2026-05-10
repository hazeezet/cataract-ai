import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Cataract Screening App", layout="centered")

st.title("Deep Learning Cataract Detection Prototype")
st.write("Upload a Retinal Fundus image to triage for Cataract presence.")

IMG_SIZE = 300  # v3 uses EfficientNet-B3 at 300x300

@st.cache_resource
def load_model(weights_path="model.pth"):
    """
    Attempts to load v3 (EfficientNet-B3) model first.
    Falls back to v2 (ResNet50) if the architecture doesn't match.
    """
    # Try v3 architecture (EfficientNet-B3)
    try:
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        st.sidebar.success("Loaded v3 (EfficientNet-B3)")
        return model, 300
    except Exception:
        pass

    # Fallback: v2 architecture (ResNet50)
    try:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        st.sidebar.info("Loaded v2 (ResNet50)")
        return model, 224
    except Exception as e:
        st.warning(f"Could not load weights from {weights_path}: {e}")
        # Return v3 with random weights as fallback
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        model.eval()
        return model, 300

model, img_size = load_model()

def apply_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def basic_transform(image, size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize
    ])
    return t(image).unsqueeze(0)

def tta_predict(model, tensor_img):
    """5-augmentation Test-Time Augmentation for robust predictions."""
    with torch.no_grad():
        p1 = torch.sigmoid(model(tensor_img)).item()
        p2 = torch.sigmoid(model(torch.flip(tensor_img, [3]))).item()
        p3 = torch.sigmoid(model(torch.flip(tensor_img, [2]))).item()
        p4 = torch.sigmoid(model(torch.flip(tensor_img, [2, 3]))).item()
        p5 = torch.sigmoid(model(torch.rot90(tensor_img, k=1, dims=[2, 3]))).item()
    return (p1 + p2 + p3 + p4 + p5) / 5.0

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Original Image", use_column_width=True)

    # Process
    if st.checkbox("Apply CLAHE Preprocessing", value=True):
        img_np = np.array(img)
        img_np = apply_clahe(img_np)
        img = Image.fromarray(img_np)
        st.image(img, caption="CLAHE Preprocessed Image", use_column_width=True)

    # Transform and Predict with Enhanced TTA
    tensor_img = basic_transform(img, img_size)
    prob = tta_predict(model, tensor_img)

    st.subheader("Results")
    if prob > 0.50:
        st.error(f"⚠️ Positive for Cataract (Confidence: {prob * 100:.2f}%)")
    else:
        st.success(f"✅ Normal (Cataract Confidence: {prob * 100:.2f}%)")
