import pandas as pd 
import numpy as np 

data = pd.read_csv('./data/iris.csv')
#NaN값 있는지 체크하기 - 각 필드별로 NaN값 있는 개수 출력
print( data.isnull().sum())

sepal_length_mean = data['sepal.length'].mean() 
sepal_width_mean = data['sepal.width'].mean() 
petal_length_mean = data['petal.length'].mean() 
petal_width_mean = data['petal.width'].mean() 

data['sepal.length'].fillna( sepal_length_mean, inplace=True)
data['sepal.width'].fillna( sepal_width_mean, inplace=True)
data['petal.length'].fillna( petal_length_mean, inplace=True)
data['petal.width'].fillna( petal_width_mean, inplace=True)

#정규화 함수 
def normalize(columnname):
    max = data[columnname].max() 
    min = data[columnname].min() 
    return (data[columnname]-min )/(max-min)

data['sepal.length'] = normalize('sepal.length')
data['sepal.width'] = normalize('sepal.width')
data['petal.length'] = normalize('petal.length')

count, bins = np.histogram( data['petal.length'], bins=3)
bin_name=["A", "B", "C"]
data['petal_grade'] = pd.cut(x = data['petal.length'], bins=bins, 
                            labels=bin_name, 
                            include_lowest=True)
print(data)


#카테고리타입분석, 분류분석, 텍스트분석
#사이킷런 fit 학습하다. 이차원배열(2d)형태로만 입력을 받는다
# 새로운 축을 추가해서 1d -> 2d   
Y_class = np.array(data['petal_grade']).reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(Y_class)

Y_class_onehot = enc.transform(Y_class).toarray()
Y_class_recovery = np.argmax(Y_class_onehot, axis=1).reshape(-1,1)

print(Y_class_onehot[:10])
print(Y_class_recovery[:10])



