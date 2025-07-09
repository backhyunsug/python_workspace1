#iris_파일.py 

import pandas as pd
import numpy as np 
df = pd.read_csv("./data/iris.csv")
print(df.head())
print(df.columns)
print(df.describe())
print(df.info())

X = df.iloc[:, :4] #전체행, 0~3 입력값 - 입력데이터 
y = df.iloc[:, 4] #4번 열만 가져온다  - 출력데이터 
print(X[:4])
print(y[:4])

#결측치제거
#이상치제거, 
#스케일링(정규화, 표준화)-서포트벡터머신, 딥러닝은 반드시 해줘야한다 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))








