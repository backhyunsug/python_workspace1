"""
범주형 자료의 경우에 어떻게 처리할것인가? 범주형자료가 문자열로 들어오는 경우도 있고
숫자형형태인경우 ( 1.대 2.중 3.소) 1 2 3 
범주형 데이터를 정확히 찾아서 범주형으로 바꿔주고 라벨링이나 원핫인코딩 
1.대 
2.중
3.소 

직업분류 1.학생 2.주부 3.직장인 4.프리랜서 5.회계사 6.변호사 7.교사 8.교수,........ 16개 정도되면 
1 보다 16이 너무 큰값이라 연산시 중요한걸로 인식된다. 

직업1  직업2 직업3.................................직업15
1      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
결과는 문자열도 알아서 처리하고 있어서 굳이 뭔가 작업 필요없음 
입력데이터는 반드시 해줘야 한다  
"""
import pandas as pd 
import mglearn 
import os 
import matplotlib.pyplot as plt 
import numpy as np 

data = pd.read_csv( os.path.join(mglearn.datasets.DATA_PATH, "adult.data"), 
                                 header=None, index_col=False, 
           names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income']
           )
print(data.head())

data = data[ ['age', 'workclass', 'education', 'gender', 'hours-per-week', 
              'occupation', 'income'] ] #마지막 필드가 타겟임 
print(data.head())
#get_dummies 함수 
data = pd.get_dummies(data)
print(data.head())
print(data.columns)

#타겟까지 넣는 바람에 타겟도 원핫인코딩 된 상태 
X = data.loc[:, 'age':'occupation_ Transport-moving'] #loc로 필드명 쓰면 그 필드명 까지 포함 
y = data.loc[:, 'income_ >50K']
print(X.head())
print(y.head())

from sklearn.linear_model import LogisticRegression 

model = LogisticRegression() 
model.fit(X, y)
print(model.score(X, y))
