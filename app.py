import streamlit as st
import cv2
import torch
from torchvision import transforms, models

from utils import detect_color, speak, save_snapshot

st.title("👕 AI Clothing Detection project")

# Load pretrained ImageNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# ImageNet label list
LABELS = {idx: label for idx, label in enumerate(open('imagenet_classes.txt').read().splitlines())}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[ 0.229, 0.224, 0.225])
])

camera = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])

if st.checkbox("Start Webcam"):
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam.")
    else:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img)

        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

        top_preds = [LABELS[p.item()] for p in pred[0][:3]]
        primary = top_preds[0]
        color = detect_color(img)

        st.success(f"{color} — {primary}")
        speak(f"You are wearing {color} {primary}")

        if st.button("📸 Save Snapshot"):
            path = save_snapshot(frame, f"{color}_{primary.replace(' ', '_')}")
            st.info(f"Snapshot saved: {path}")
else:
    camera.release()
