import os
import shutil
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 경로 설정 및 하이퍼파라미터
original_dir = pathlib.Path("./dogs-vs-cats/train")
new_base_dir = pathlib.Path("./dogs-vs-cats/cats_vs_dogs_small")

batch_size = 32
img_height = 180
img_width = 180
num_epochs = 3 # TensorFlow 예제와 동일하게 설정
model_save_path = "convnet_from_scratch.pth"

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 2. 데이터셋 디렉터리 생성 및 이미지 복사
def make_subset(subset_name, start_index, end_index):
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir, exist_ok=True)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

if not os.path.exists(new_base_dir):
    print("데이터셋 서브셋 생성 중...")
    make_subset("train", start_index=0, end_index=1000)
    make_subset("validation", start_index=1000, end_index=1500)
    make_subset("test", start_index=1500, end_index=2500)
    print("데이터셋 서브셋 생성 완료.")

# 3. 데이터 증강 및 전처리
# PyTorch의 transforms를 사용하여 데이터 증강 파이프라인 구축
train_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)), # RandomZoom 대신 사용
    transforms.ToTensor() # 이미지를 [0, 1] 범위의 텐서로 변환
])

val_test_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

# 4. 데이터셋 로드 및 데이터로더 생성
train_dataset = datasets.ImageFolder(new_base_dir / "train", transform=train_transforms)
validation_dataset = datasets.ImageFolder(new_base_dir / "validation", transform=val_test_transforms)
test_dataset = datasets.ImageFolder(new_base_dir / "test", transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 5. 모델 정의
# Keras 함수형 API와 유사하게 계층을 쌓는 방식
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU()
        )
        # 풀링 후 이미지 크기 계산: 180/2 -> 90/2 -> 45/2 -> 22.5 -> 22/2 -> 11
        # Keras와 동일한 구조를 위해 Resizing을 추가함
        # 180x180 -> 90x90 -> 45x45 -> 22x22 -> 11x11
        # 정확한 크기는 계산 후 확인 필요
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dense(256 * 11 * 11, 1), # Flatten된 크기 계산 필요
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # TensorFlow의 Rescaling(1./255)는 ToTensor()가 이미 수행함
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

model = ConvNet().to(device)
print(model)

# 6. 모델 컴파일 (옵티마이저, 손실 함수 정의)
optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
criterion = nn.BCELoss() # Binary Cross-Entropy Loss

# 7. 모델 학습
best_val_loss = float('inf')
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    # 훈련 단계
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    # 검증 단계
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            predicted = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    epoch_val_loss = val_loss / len(validation_loader.dataset)
    epoch_val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {epoch_loss:.4f}, train_acc: {epoch_acc:.4f} - val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}")
    
    # 히스토리 기록
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_acc)

    # Keras의 ModelCheckpoint 구현
    if epoch_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}. Saving model...")
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), model_save_path)

# 8. 그래프 그리기 (TensorFlow 예제와 동일)
accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]
loss = history["loss"]
val_loss = history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# 9. 모델 로드 및 테스트
# PyTorch에서는 model.state_dict()를 로드
test_model = ConvNet().to(device)
test_model.load_state_dict(torch.load(model_save_path))
test_model.eval()

test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)
        
        outputs = test_model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        
        predicted = (outputs > 0.5).float()
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = test_correct / test_total
print(f"테스트 정확도: {test_acc:.3f}")