import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 데이터 불러오기 
iris = load_iris() 
X = iris.data 
y = iris.target 

#2.표준화 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#3.학습데이터와 테스트데이터 쪼개기 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#numpy 배열 -> PyTorch 타입 배열(텐서)로 변환하자 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long )
y_test_tensor = torch.tensor(y_test, dtype=torch.long )

#모델한테 전달하기 위해서 데이터 셋을 만들고 DataLoader에게 전달해야 한다 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset =  TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) #

#4.모델정의 -> 클래스로 만든다. nn.Module 라는 클래스를 반드시 상속받아야 한다. 
# 부모클래스가 하는일이 많을때 이런식으로 설계를 한다. 
class IrisClassifier(nn.Module):
    def __init__(self): #생성자 
        super(IrisClassifier, self).__init__() #부모 생성자를 호출한다. 
                                               #부모 생성자 호출코드는 메서드의 젤 처음에 와야 한다 
                                               #super 가 부모를 뜻함. 두개의 매개변수를 전달한다 
        #입력은닉층 (iris 는 4개의 특성을 갖는다)
        self.fc1 = nn.Linear(4, 16)  #fc1은 그냥 변수임 모델을 저장해 둔다. nn.Linear(입력개수, 출력개수)
        self.fc2 = nn.Linear(16, 8)  #은닉층
        self.fc3 = nn.Linear(8, 3)   #출력층( 출력결과가 3이어야 한다. ) 파이토치는 softmax함수 안쓴다 
                                      #손실함수에서 softmax함수가 호출된다. 
        self.active = nn.ReLU() #활성화 함수 - relu 

    def forward(self, x):
        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
        x = self.active(x)
        x = self.fc3(x)
        return x

#5. 모델만들고, 손실함수, 옵티마이저 

model = IrisClassifier() 
criterion = nn.CrossEntropyLoss()  #손실함수 - 다중분류, 소프트맥스를 제공함 
optimizer = optim.Adam(model.parameters(), lr=0.01)  #옵티마이저 

#6.학습 
def train_model(epochs):
    for epoch in range(epochs):
        #train_loader - 배치사이즈만큼 
        for inputs, labels in train_loader:  #현재 배치사이즈 16개임, 16개씩 가져온다 
            optimizer.zero_grad() #옵티마이저 초기화 
            outputs = model(inputs)  #순전파, 가중치 계산중 
            loss = criterion(outputs, labels) #손실값을 계산한다. 
            loss.backward() #오차의 역전파 
            optimizer.step()  
        print(f'Epochs[ {epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("학습완료")


def evaluate_model(): 
    #모델을 평가모드로 변경한다 
    model.eval()

    with torch.no_grad(): #그라디언트 계산 비활성화 
        correct_train =0 #훈련셋이 예측이 잘 맞는경우  카운트하기 위한 변수 
        total_train = 0 #전체 개수 
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  #출력값이 확률, np.argmax쓰듯이 젤 확룰 높은 찾기
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item() 
            #현재 16개씩 처리하고 있음 
        accuracy_train = 100*correct_train/total_train
        print(f"훈련셋 정확도 :  {accuracy_train}")

    with torch.no_grad(): #그라디언트 계산 비활성화 
        correct_train =0 #훈련셋이 예측이 잘 맞는경우  카운트하기 위한 변수 
        total_train = 0 #전체 개수 
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  #출력값이 확률, np.argmax쓰듯이 젤 확룰 높은 찾기
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item() 
            #현재 16개씩 처리하고 있음 
        accuracy_train = 100*correct_train/total_train
        print(f"테스트셋 정확도 :  {accuracy_train}")


if __name__ == "__main__":
    train_model(epochs=100)
    evaluate_model()


















             
        





