import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
import pandas as pd 
# 1. 데이터 준비

diamond = pd.read_csv("./data/diamonds.csv")
print(diamond.head())
print(diamond.info())
print(diamond.describe())


# # 결측값 및 불필요한 컬럼 처리 (예: 'horsepower' 컬럼에 ?가 포함된 경우)
# # 먼저 'horsepower' 컬럼의 문자열 '?'를 NaN으로 변환하고, 해당 행을 제거
# diamond = diamond.dropna(how="any", axis=0)

# print(diamond.info())

print("-------------------")
X = diamond.iloc[:, [0,1,2,3,4,5,7,8,9]]
print(X.head())
y = diamond.loc[:, 'price']
print(y[:10])

#원핫인코딩 
X = pd.get_dummies(X)
print(X.head())
print(X.shape) #특성의 개수 알아내기 


#데이터 표준화 
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#stratify - 라벨이 불균형할때 그 비율에 맞춰서 나눠라 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) #회귀나 2진분류일경우 1차원임 => 2차원

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader( train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader( test_dataset,  batch_size=32, shuffle=False)


class HousingClassifier(nn.Module):
    def __init__(self):
        super(HousingClassifier, self).__init__() 
        self.input = nn.Linear(26, 64) #특성 8 
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1) #결과가 하나임 

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return x 
        
model = HousingClassifier()
criterion = nn.MSELoss() #Mean squared Error  
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


#평가는 다르게 
def evaluate_model():
    model.eval() #평가모드 
    with torch.no_grad():
        total_loss = 0 
        total_samples  = 0 
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)  #반올림이라 0 아니면 1임
            total_samples +=inputs.size(0)
            total_loss+= loss.item()*inputs.size(0)
            
        #MSE와 RMSE계산
        avg_mse = total_loss / total_samples 
        rmse = np.sqrt(avg_mse)
        print(f'테스트 데이터셋 평균 MSE :  {avg_mse:.4f}')
        print(f'테스트 데이터셋 RMSE    :  {rmse:.4f}')
        
if __name__ == "__main__":
    train_model(1000)
    evaluate_model()

