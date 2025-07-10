"""
PCA(주성분분석) - 차원축소 
특성이 너무 많을때 각 특성들로부터 새로운 특성을 만들어낸다. 
다중공선성 : 댐 A1, A2 ,B1, C1, ....       target 수량에 
단일선형회귀 :X가 하나임 y한테 영향을 미치는 요소가 X하나임 기울기 절편 구하면 
다중선형회귀 :X들간에 상호 연관관계가 있어서 제거시키는게 나을때가 있음 
            PCA가 이런부분들까지 알아서 새로운 요소를 만든다. 
            각각의 분산을 뽑아서 분산이 서로 최대가 되는 방향으로 회전을 시키고 
            여러가지 조작을 가해서 새로운 특성을 뽑아낸다. 
            전체 특성을 재배열해서 새로운 특성을 만들어낸다. - 특성의 개수 지정가능 
30 => 2 차원축소 
   => 3차원 이상은 시각화가 아려운데 시각화 하기 좋다 
     노이즈(잡음) 제거 
     과적합방지 기본적으로 특성이 너무 많으면 과적합이 일어나기 쉽다. 일반화에 이바지
     계산속도도 빨라진다.
     fit, transform => PCA뽑아내고 다시 학습한다. PCA자료로
"""

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt #이전 버전에서는 pyplot가 반드시 먼저 import가 되어야 한다
import seaborn as sns   #seaborn도 pyplot기반이라서 
import pandas as pd 

#cancer = load_breast_cancer()
cancer = load_digits() 
X = cancer['data']
y = cancer['target']

print(X.shape)
#스케일링 
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler() 
scalar.fit(X)
X_scaled = scalar.transform(X)  

#주성분분석
from sklearn.decomposition import PCA 
pca = PCA(n_components=10) #성분개수 지정하기 
pca.fit(X_scaled) #학습 
X_pca = pca.transform(X_scaled)

X_train, X_test, y_train, y_test=train_test_split(X, y, random_state=0)
X_train_scaled, X_test_scaled, y_train, \
    y_test = train_test_split(X_scaled, y, random_state=0)
X_train_pca, X_test_pca, y_train, \
    y_test = train_test_split(X_pca, y, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("-------- 기본 ---------")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

model.fit(X_train_scaled, y_train)
print("-------- 스케일링 ---------")
print("훈련셋", model.score(X_train_scaled, y_train))
print("테스트셋", model.score(X_test_scaled, y_test))

model.fit(X_train_pca, y_train)
print("-------- PCA ---------")
print("훈련셋", model.score(X_train_pca, y_train))
print("테스트셋", model.score(X_test_pca, y_test))

