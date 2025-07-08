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

#의사결정트리 -> 회귀 가능, 트리계열은 언제나 과대적합상태임  
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== DecisionTreeRegressor Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test)) #회귀분석에서 score 가 결정계수값이 나오는데 음수면 위험 
print("특성의 중요도 ", model.feature_importances_)

#랜덤포레스트 : 의사결정트리 + 업그레이드, 여러개의 분석기를 함께 사용 - 앙상블
#트리를 랜덤하게 많이 만들어서 평균값을 구한다. 할때마다 별도의 트리가 만들어져서 계속 측정치가 달라진다. 
#  n_estimators : 만들 트리 최대 개수 
# max_depth : 트리의 최대 깊이 지정 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=0, n_estimators=300, max_depth=3)
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== RandomForestRegressor Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test)) #회귀분석에서 score 가 결정계수값이 나오는데 음수면 위험 
print("특성의 중요도 ", model.feature_importances_)

#그라디언트부스팅 : 앙상블 계열, 약한학습기들을 통해서 학습을 하고 보정작업을 거쳐서 결과를 찾아낸다.
#sklearn GradientBoostion,  xgboost 라이브러리,  LightGBM ..... 
#  

#conda install xgboost 
#conda install LightGBM  , Microsoft Visual C++ Build Tools  설치필요

#learning_rate=0.1 학습률 , 머신러닝이 학습하는 속도를 조절한다. 
#너무 높으면 : 막 빨리 빨리 학습하다가 잘못해서 최적의 위치를 지나쳐 갈 수 있다 
#너무 낮으면 : 아주 천천히 느리게 학습을 한다. 아무리 가도 최저점을 못 도달하는 경우도 
#GridSearch : 하이퍼파라미터들을 주면 알아서 테스트를 하면서 적절한 파라미터를 찾아낸다. 
#            오래걸림 

from sklearn.ensemble import GradientBoostingRegressor 
model = GradientBoostingRegressor(random_state=0, n_estimators=10, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== GradientBoostingRegressor Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test)) #회귀분석에서 score 가 결정계수값이 나오는데 음수면 위험 
print("특성의 중요도 ", model.feature_importances_)

#xgboost 
from xgboost import XGBRegressor 
model = XGBRegressor(random_state=0, n_estimators=10, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train) #학습을하고
y_pred = model.predict(X_test)
#선형회귀모델의 score함수 썼을때 결정계수 1이면 완벽하게 예측을 한거고 
#0이면 거의 예측불가 - 인 경우가 있는데 심각하게 안맞는 상태 
print("=== XGBRegressor Model ===")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test)) #회귀분석에서 score 가 결정계수값이 나오는데 음수면 위험 
print("특성의 중요도 ", model.feature_importances_)


