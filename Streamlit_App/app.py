import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import json

# Load the trained model
# Note: Replace 'LeafNet.pth' with the actual model path once available
@st.cache_resource
def load_model():
    model = torch.load("LeafNet.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

# Define preprocessing steps
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Load class labels
def load_class_mapping():
    with open("class_mapping.json", "r") as f:
        return json.load(f)

# Load parenting suggestions
def load_suggestions():
    with open("suggestions.json", "r") as f:
        return json.load(f)

# Inference function
def predict(image, model, class_map, suggestions):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = class_map[str(predicted.item())]
        advice = suggestions.get(class_name, "No suggestion available.")
    return class_name, advice

# Streamlit UI
st.title("🌿 Digital Pheno-Parenting")
st.subheader("Plant Species & Disease Detection with Care Suggestions")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")

    try:
        model = load_model()
        class_map = load_class_mapping()
        suggestions = load_suggestions()

        prediction, advice = predict(image, model, class_map, suggestions)

        st.success(f"Prediction: {prediction}")
        st.info(f"Care Suggestion: {advice}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")