# 입력데이터 [[1],[2],[3],,,,,,,,,,, ]
# 출력데이터 [0,0,0,1,0,0,0,1 ............. ]  
# 1~2025년까지 
def isLeap(year):
    if year%4==0 and  year%100!=0 or year%400==0:
        return 1
    return 0 

X = [] 
y = [] 
for i in range(1, 2026):
    X.append(i)
    y.append(isLeap(i))
print(X[2000:])
print(y[2000:])
#머신러닝, X는 반드시 ndarray(2d), y는 1d array 
import numpy as np 
X = np.array(X)
#차원하나를 추가해서 2차원으로 만들자 
X = X.reshape(-1, 1) 
print(X.shape)  #2d array 타입으로 만들어야 한다 
y = np.array(y)

#1.훈련셋과 테스트셋을 나누자 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3) #Knn 이웃 분류알고리즘 
model.fit(X_train, y_train) #X, y 값 갖고 학습을 진행한다.
print("훈련셋 : ", model.score(X_train, y_train))
print("테스트셋 : ", model.score(X_test, y_test))

