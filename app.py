import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. 모델 및 클래스 설정 (train.py에서 저장한 파일 활용)
@st.cache_resource
def load_model():
    with open("class_names.txt", "r") as f:
        class_names = f.read().splitlines()
        
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("best_pokemon_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

model, class_names = load_model()

# 2. 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Streamlit UI 구성
st.title("Pokédex: Pokemon Classifier ⚡")
st.write("포켓몬 이미지를 업로드하면 이름을 예측합니다! (Top-5 Predictions)")

uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 포켓몬', use_container_width=True)
    
    st.write("분석 중...")
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Top 5 추출
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    st.subheader("Top-5 Predictions:")
    for i in range(top5_prob.size(0)):
        name = class_names[top5_catid[i]]
        prob = top5_prob[i].item() * 100
        st.write(f"{i+1}. **{name}** ({prob:.2f}%)")
        st.progress(prob / 100)