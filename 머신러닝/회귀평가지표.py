
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing # 캘리포니아 집값 데이터셋
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # 특성 스케일링 (선형 모델에 중요)

print("--- 캘리포니아 집값 회귀 분석 및 평가 예제 시작 ---")

# 1. 캘리포니아 집값 데이터셋 로드
housing = fetch_california_housing()
print(type(housing)) #딥러닝(Tensorflow:c++)->케라스(초보자용),파이토치(세밀하게 제어가 가능하다)

#차트그리려면 numpy -> 요소 하나씩 그려야 해서 차트를 그리고싶다. 
#DataFrame :데이타프레임 자체가 차트를 제공하기도 하고 seaborn 차트 , plotly차트(interactive차트)
#python 코드로 차트를 그리면 plotly 는 html과 css와 자바스크립트로 움직이는 차트가 만들어진다. 
#예전에는 R언어만 지원했는데 현재 Python 라이브러리가 지원을 하고 있다  
#넘파이배열, 컬럼명 
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="houseval") 

print(X.head())
print(X.columns)
print(y[:10])

#1.훈련셋 쪼개기 - numpy 2차원(dataframe), 1차원(series),
X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2.결측치,이상치 없음 
#3.스케일링 또는 표준화 정규화
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) #비지도학습 fit -> transform  
X_test_scaled = scaler.fit_transform(X_test)  

#4.선형회귀모델학습
model = LinearRegression() 
model.fit(X_train_scaled, y_train)

#5.테스트데이터로 예측하기 
y_pred = model.predict(X_test_scaled)

#6.회귀모델 평가지표    -    실제값, 기대값=예측값        
mae = mean_absolute_error(y_test, y_pred)
print("mae:평균절대오차", mae)

mse = mean_squared_error(y_test, y_pred)
print("mse:평균제곱오차", mse)

rmse = np.sqrt(mse)
print("mse:평균제곱근오차", rmse)

#결정계수가 1에 가까울 수록 좋다. 다만 특성의 개수가 많아지면 결정계수값이 예측력하고 상관없이 좋아진다. 
#특성이 많아질경우에는 mae나 mse를 쓰자. mse가 이상치에 영향을 많이 받는다 
 
r2 = r2_score(y_test, y_pred) #결정계수, 평상시 score 함수와 동일 
print("결정계수 : ", r2)

print("모델의 score ", model.score(X_test_scaled, y_test))

#차트를 그릴려고 -> 특성이 몇개 안되서 산포도행렬을 그릴려고, 데이터 개수자체가 너무 많아도 문제가 된다.  

print("데이터개수")
print(X.shape)

X["hosuingval"] = y 
print(X.head())
#데이터가 2만개가 넘는다. 차트를 그려보고 , 데이터샘플링 
df_sample = X.sample(n=2000, random_state=42) #데이터를 원하는 만큼 샘플링을 해온다
print(df_sample.shape)

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.pairplot(df_sample, 
             diag_kind='kde',  #대각선 히스트그램 밀도 부드럽게
             kind='scatter')
plt.show() 
















