#iris_파일.py 

import pandas as pd
import numpy as np 
df = pd.read_csv("./data/penguins.csv")
print(df.head())
print(df.columns)
print(df.describe())
print(df.info())

#라벨인코딩 작업 island 연산이 불가 => 연산이 가능하게 도와준다. 
df.loc[df["island"] == "Torgersen", "island"] = 1
df.loc[df["island"] == "Dream",     "island"] = 2
df.loc[df["island"] == "Biscoe",    "island"] = 3

#성별 
df.loc[df["sex"] == "MALE", "sex"] = 1
df.loc[df["sex"] == "FEMALE", "sex"] = 2
#map과 유사하게 apply, 함수만들어서 모든 요소한테 그 함수를 적용해라  
print(df.head(10))

#결측치 - 중간에 있을 수도 있으니까 
print(df.isna.sum())
#결측치 - 제거, 대부분의 경우는 제거보다는 다른값으로 대체하는경우가 많다. 
df = df.dropna(how="any", axis=0) #NaN 값이 있는 행은 모두 삭제해라 

print("결측치 삭제이후")
print(df.isna().sum())
print("------------")
print(df.shape)
print(df.shape)

X = df.iloc[:, 1:] #전체행, 나머지가 입력데이터, 특성, 픽처 
y = df.iloc[:,  0] #0번열이 목표치 라벨
print(X[:4])
print(y[:4])

#이상치제거, 
#스케일링(정규화, 표준화)-서포트벡터머신, 딥러닝은 반드시 해줘야한다 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))








