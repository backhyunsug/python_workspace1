#ColumnTransformer 라는 클래스가 있다. 
#컬럼단위로 전처리 작업을 해야 할때 쭈욱 지정해놓으면 이것저것 적용을 해준다 

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

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler 

st = ColumnTransformer( [
    ("scaling", StandardScaler(), ['age', 'hours-per-week']),
    ("onehot", OneHotEncoder(sparse_output=False), [
        'workclass', 'education', 'gender', 'occupation'
    ])
])

st.fit(data)
print(st.transform(data))
