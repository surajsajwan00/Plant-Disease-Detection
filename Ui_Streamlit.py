import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_diseases):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_ftrs, num_diseases)

    def forward(self, x):
        x = self.vgg(x)
        return x

## Load model
num_diseases = 38 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(num_diseases).to(device)
model.load_state_dict(torch.load('Model/Plant_Disease_Detection_Model_VGG.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

st.title("Plant Disease Detection")
st.write("Upload an image to classify plant diseases.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
    
## Class labels
    class_labels = [
        'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
        'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch',
        'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
        'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
        'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy',
        'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
        'Raspberry___healthy', 'Tomato___Leaf_Mold',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot',
        'Corn_(maize)___healthy'
    ]
    
    st.write(f"Prediction: {class_labels[class_idx]}")