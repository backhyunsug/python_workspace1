#당뇨병과 관련된 요소들이 있음 , 1년뒤에 값들을 예측해야 한다.
#알고리즘 중에 Knn이웃, 의사결정트리, 랜덤포레스트등 ...  분류뿐만 아니라 회귀도 지원한다. 
#Ridge, Lasso   
from sklearn.datasets import load_diabetes  
data = load_diabetes() #bunch 라는 클래스 타입으로 정리해서 준다
#이상치, 누락치, 정규화 까지 다 된 자료를 준다 - pandas, numpy
#  
print(data.keys()) 
print(data["target"][:10])
print(data["data"][:10])
print(data["DESCR"])

X = data["data"] #현재 10개의 특성값이 
y = data["target"]# 미래값으로 나타나는거 

print(X.shape) #442개이고 특성이 10개임 
print(y.shape)

#데이터를 나눈다 
from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split  \
(X, y, random_state=1234)  #7.5:2.5 비율로 나뉜다 

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== Linear Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

from sklearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== Ridge Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== Lasso Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))






