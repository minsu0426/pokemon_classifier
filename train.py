import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import kagglehub
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Kaggle 데이터셋 다운로드
print("Downloading dataset...")
data_path = kagglehub.dataset_download("lantian773030/pokemonclassification")

dataset_dir = os.path.join(data_path, "PokemonData") 

# 2. 데이터 전처리 및 로더 설정
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3. 모델 설정 함수
def get_model(model_name, pretrained, fine_tune_all):
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if not fine_tune_all:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        if not fine_tune_all:
            for param in model.features.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    return model.to(device)

# 4. 학습 및 평가 함수
def train_and_evaluate(model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

    # 평가
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Precision, Recall, F1 계산
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = sum(1 for x,y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    
    return accuracy, precision, recall

# 5. 4가지 실험 설정 정의
experiments = {
    "Exp 1: ResNet18 (Pretrained, Feature Extractor)": get_model('resnet18', True, False),
    "Exp 2: ResNet18 (Pretrained, Fine-tune All)": get_model('resnet18', True, True),
    "Exp 3: VGG16 (Pretrained, Feature Extractor)": get_model('vgg16', True, False),
    "Exp 4: ResNet18 (Scratch, Train All)": get_model('resnet18', False, True)
}

# 6. 실험 실행 및 결과 저장
if __name__ == '__main__':

    results = {}
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting Experiments ---")
    for exp_name, model in experiments.items():
        print(f"\nRunning {exp_name}...")
        
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_update, lr=0.001)
        
        acc, prec, rec = train_and_evaluate(model, criterion, optimizer, num_epochs=10)
        results[exp_name] = {"Accuracy": acc, "Precision": prec, "Recall": rec}
        print(f"Result -> Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}")
        
        if "Fine-tune All" in exp_name:
            torch.save(model.state_dict(), "best_pokemon_model.pth")
            with open("class_names.txt", "w") as f:
                f.write("\n".join(class_names))

    print("\n--- Final Results ---")
    for exp, metrics in results.items():
        print(f"{exp}: {metrics}")