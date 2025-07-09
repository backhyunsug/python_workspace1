#iris_파일.py 
#10개미만 
#텍스트?  
import pandas as pd
import numpy as np 
df = pd.read_csv("./data/diamonds.csv")
print(df.head())
print(df.columns)
print(df.describe())
print(df.info())
#1. cut 나 clarity 의 경우면 분류 
#2.price 의 경우는 회귀로 
# cut color clarity -> 카테고리화 시켜서 -> 라벨인코딩을 하거나 원핫인코딩하기 
# value_counts()->속성하고 개수
print("cutting") 
print(df["cut"].value_counts()) 
print("color") 
print(df["color"].value_counts()) 
print("clarity") 
print(df["clarity"].value_counts()) 
print(df["color"].unique()) #특정필드에서 값 하나씩만 가져온다 

def getLabelMap(field):
    colorList = df[field].unique()
    #list를 받아와서 map으로 바꾸기  
    colorMap = {item: index + 1 for index, item in enumerate(colorList)}
    # for key, value in colorMap.items():
    #     print(key, value)
    return colorMap 

def changeLabeling():
    colorMap = getLabelMap('color')
    df['color_label'] = df['color'].map(colorMap)
    colorMap = getLabelMap('clarity')
    df['clarity_label'] = df['clarity'].map(colorMap)
    colorMap = getLabelMap('cut')
    df['cut_label'] = df['cut'].map(colorMap)

changeLabeling()
print(df.head())

X = df.loc[:, ['carat', 'depth', 'table', 'price', 'x', 'y',
       'z','color_label', 'clarity_label']]

y = df.loc[:, 'cut_label']
print(X[:5])
print(y[:5])

# #결측치 - 중간에 있을 수도 있으니까 
# print(df.isna.sum())
# #결측치 - 제거, 대부분의 경우는 제거보다는 다른값으로 대체하는경우가 많다. 
# df = df.dropna(how="any", axis=0) #NaN 값이 있는 행은 모두 삭제해라 

# print("결측치 삭제이후")
# print(df.isna().sum())
# print("------------")
# print(df.shape)
# print(df.shape)

# X = df.iloc[:, 1:] #전체행, 나머지가 입력데이터, 특성, 픽처 
# y = df.iloc[:,  0] #0번열이 목표치 라벨
# print(X[:4])
# print(y[:4])

# #이상치제거, 
# #스케일링(정규화, 표준화)-서포트벡터머신, 딥러닝은 반드시 해줘야한다 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0 )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))








