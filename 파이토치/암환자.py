import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

cancer = load_breast_cancer()
X = cancer.data 
y = cancer.target 

#데이터 표준화 
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#stratify - 라벨이 불균형할때 그 비율에 맞춰서 나눠라 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) #2진분류일때 shape변경

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader( test_dataset,  batch_size=32, shuffle=False)


class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__() 
        self.fc1 = nn.Linear(30, 64) #특성 30 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1) #결과가 하나임 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x 
        
model = CancerClassifier()
criterion = nn.BCEWithLogitsLoss() #시그모이드  + Binary Cross Entropy 
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=100):
    model.train() #학습모드 
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.step() 
        print(f'Epoch {epoch+1}/{epochs}, Loss :{loss.item()}')
    
    print("학습완료")



def evaluate_model():
    model.eval() #평가모드 
    with torch.no_grad():
        correct = 0 
        total = 0 
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))  #반올림이라 0 아니면 1임
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100*correct/total
        print(f'테스트셋 {accuracy}')
    

if __name__ == "__main__":
    train_model(100)
    evaluate_model()

