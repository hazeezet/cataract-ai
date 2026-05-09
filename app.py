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

@st.cache_resource
def load_model(weights_path="model.pth"):
    # Reconstruct the ResNet50 custom head from Phase 3
    model = models.resnet50(pretrained=False) # instantiate randomly
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    # Load state dict map
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.warning(f"Could not load weights from {weights_path}. Model predictions will be random.")
    
    model.eval()
    return model

model = load_model()

def apply_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def basic_transform(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return t(image).unsqueeze(0) # add batch dim

uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Original Image", use_column_width=True)

    # Process
    if st.checkbox("Apply CLAHE Preprocessing"):
        img_np = np.array(img)
        img_np = apply_clahe(img_np)
        img = Image.fromarray(img_np)
        st.image(img, caption="CLAHE Preprocessed Image", use_column_width=True)

    # Transform and Predict
    tensor_img = basic_transform(img)

    with torch.no_grad():
        output = model(tensor_img)
        prob = torch.sigmoid(output).item()
    
    st.subheader("Results")
    if prob > 0.5:
        st.error(f"⚠️ Positive for Cataract (Confidence: {prob * 100:.2f}%)")
    else:
        st.success(f"✅ Normal (Cataract Confidence: {prob * 100:.2f}%)")

