from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()  #Bunch타입 
print(cancer.keys())
X = cancer['data']
y = cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""
대부분의 머신러닝 알고리즘 평면에 선을 긋는다 데이터에 따라서는 평면에 선을 못긋는 경우에 
수학자 차원을 분리시켜서 평면의 다차원공간으로 보내서 차원간에 선을 긋는다 
"""
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()#분류
model.fit(X_train, y_train)
print("-------- 로지스틱 --------------")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

from sklearn.svm import SVC
model = SVC()#분류
model.fit(X_train, y_train)
print("-------- 스케일링 안한 서포트벡터머신 ------------")
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

