
url="http://lib.stat.cmu.edu/datasets/boston"
#다중회귀분석 : 공분산(특성간에 서로 영향을 주고받는것)을 따져보고 필요없는 특성은 제거하는게 맞다 
#R은 기본적으로 제거해준다. 파이썬은 우리가 해야 한다.
#powershell 이 파일명에 () 를 안받아준다  
import pandas as pd   #다양한 유형의 데이터가 있을때 처리 방법 
import numpy as np     
#분리문자가 공백이 아니고 \s+, skiprows = 22줄 건너뛰기  제목줄이 없다. 
df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
print(df.head(10))

#np 에 hstack 함수가 있음, 수평방향으로 배열을 이어붙이는 함수 
#짝수행에 홀수를 갖다가 붙인다. df.values[::2, : ] 0,2,4,6,8  : 전체컬럼 
#홀수행에 앞에 열 2개만 df.values[1::2, :2] 
X = np.hstack( [df.values[::2, : ],  df.values[1::2, :2]] )
print(X.shape, X[:10])  
y = df.values[1::2, 2] #이 열이 target
print(y.shape) 
#행의 개수가 같아야 머신러닝 연산을 수행한다. 결과가 입력한 데이터 개수만큼 있어야 한다 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.linear_model import LinearRegression 
model = LinearRegression() 
model.fit(X_train, y_train)
print("LinearRegression")
print("훈련셋 " , model.score(X_train, y_train))
print("테스트셋 " , model.score(X_test, y_test))

print("기울기들", model.coef_)
print("절편", model.intercept_)

#선형회귀분석: 다중공선성문제, 여러 특성간에 서로 너무 밀접해서 필요없는 요소들 이걸 고려하지를 않는다 
#특성의 개수가 많을경우에 처리능력이 떨어진다. 
#보스톤 주택가격데이터의 특성의 개수는 13개임, 즉 가중치도 13개가 나온다. 

#누군가가 가중치를 규제하면 과대적합을 막을 수 있지 않을까?
#가중치가 너무 크거나 작으면 훈련데이터셋에 초점이 맞춰진다. 과대적합 
#가중치를 규제 하자 라쏘는 가중치를 0에 가깝게 하다가 불필요한 요소가 있으면 아예 0으로 만들기도 한다. 
#모델을 심플하게 만든다. L2 정규화 
#리지는 계수를 완전히 0으로 만들지는 못한다. L1 정규화 
#하이퍼파라미터 alpha 라는 값이 있는데 이걸 0으로 놓으면 규제를 아무것도 안하겠다. LinearRegression하고 
#똑같음 alpha를 키우면 규제가 높아진다. 적절한 alpha를 찾는게 일임 

#라쏘하고 리지라는 알고리즘이 있다. 
#라쏘는 쓸데없는 계수(기울기들)

from sklearn.linear_model import Ridge 
model = Ridge(alpha=10) 
model.fit(X_train, y_train)
print("Ridge")
print("훈련셋 " , model.score(X_train, y_train))
print("테스트셋 " , model.score(X_test, y_test))
print("기울기들", model.coef_)
print("절편", model.intercept_)      

from sklearn.linear_model import Lasso 
model = Lasso(alpha=10)  #숫자가 커질수록 규제가 커진다 
model.fit(X_train, y_train)
print("Lasso")
print("훈련셋 " , model.score(X_train, y_train))
print("테스트셋 " , model.score(X_test, y_test))
print("기울기들", model.coef_)
print("절편", model.intercept_)      

#alpha 값이 적절해야 한다. 
#머신러닝 알고리즘은 하이퍼파라미터 => 우리가 수작업으로 찾아야 한다. 
#딥러닝은 자동으로 찾는다. 이미지, 흑백사진의 경우는 머신러닝이나 딥러닝이나 비슷   