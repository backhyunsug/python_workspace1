#sklearn에서 load_digits() -> 손으로 쓴 숫자 맞추기 
#미국의 우편번호 나누기때 수집된 데이터 
#이미지 => 디지탈화 하는 과정에 흑백은 2차원배열, 칼라는 3차원배열임 
#이미지가 10장이 있고 각 이미지 크기가 150 by 150 
#10 150*150이 특성의 개수가 된다.  이미지 => numpy배열로 바꿔야 : 파이썬이 PIL 라이브러리로 이걸 제공
#  

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. 데이터 준비 ---
data = load_digits()
X = data['data']
y = data['target']
print(X.shape) 
print(X[:10])

#이미지라고 
print( data.images[:10] )#numpy 2d -> 1d 로 바꾸어서 data로 준거고, 원래데이터 
images = data.images 

# plt.figure(figsize=(10, 4)) #차트의 크기 
# plt.imshow(images[0], cmap="gray_r") #회색으로 이미지 출력 
# plt.show() 

def drawNumbers():
    #이미지를 여러개 출력하려면 화면 분할을 해야 한다.
    plt.figure(figsize=(10, 4)) #화면 전체 크기를 지정하고나서 작게 나누려면 subplot 함수를 사용한다. 
    # 2 by 5 쪼개면 10개의 화면이 만들어지고 각분할위치에 번호가 붙는다 
    # 1 2 3 4 5 
    # 6 7 8 9 10 
    for i in range(10):       
        plt.subplot(2, 5, i+1) #내가 내보낼 위치 지정 
        plt.imshow(images[i], cmap="gray_r", interpolation='nearest') #interpolation:보간법 
        plt.title(f"Label:{y[i]}")
        plt.axis('off') #축 없애기 

    plt.tight_layout() #이쁘게 다시 정리해라 
    plt.suptitle("Digits Image", y=1.05, fontsize=16) 
    #y는 제목이 출력될 위치를 말하는데 y=0 이면 아래쪽 y=1 이면 위쪽  1.05 는 영역 바깥쪽에 놓으라는 의미임 
    plt.show()

#이미지 크기 자체는 8 by 8 -> 1차원으로 바꾸니까 64개의 특성이 되었다. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

#로지스틱 
from sklearn.linear_model import LogisticRegression #분류 
model = LogisticRegression(solver='liblinear', 
                           multi_class='auto', 
                           max_iter=5000, 
                           random_state=0)
#solver - 모델계수를 찾아가는 방법 , 보통 데이터셋이 적을때는 liblinear를 써라 
#multi_class='auto'- 다중분류일때 
#max_iter=5000, 계수 찾아갈때 학습을 얼마나 반복할까를 지정하는것 
#  
model.fit(X_train, y_train)
print("로지스틱회귀")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))


# Knn이웃 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("knn 이웃")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

# 의사결정트리 
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
print("DecisionTreeClassifier")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=0)
model.fit(X_train, y_train)
print("RandomForestClassifier")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

# 그라디언트부스팅  
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=4, n_estimators=100, random_state=0, 
                                   learning_rate=0.1  )
model.fit(X_train, y_train)
print("GradientBoostingClassifier")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))





