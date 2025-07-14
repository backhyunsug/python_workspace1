import pandas as pd 
import numpy as np 

#특성이 원래는 카테고리인데 숫자형태로 입력될때 어떻게 처리할지?  
demo_df = pd.DataFrame( { 
    '숫자특성':[0,1,2,1],  #범주형
    '범주형특성': ['양말', '여우', '양말', '상자']    
})
# get_dummies : 숫자형의 범주데이터를 범주로 파악을 못함 
df1 = pd.get_dummies(demo_df)
print(df1) 
#첫번째 특성에 대해서는 그냥 숫자로 받아들이고 두번째 특성은 문자열의 경우만 범주로 받음 
#첫번째 특성을 문자열 또는 카테고리형으로 바꾸자 
# demo_df['숫자특성'] = demo_df['숫자특성'].astype(str)
# df1 = pd.get_dummies(demo_df)
# print(df1) 

print("------------------------------------")
#OneHotEncoder 클래스를 사용하자 
from sklearn.preprocessing import OneHotEncoder 
#sparse_output=False , 희소행렬이거나 numpy배열이거나, False일경우 Numpy배열 반환 
ohe = OneHotEncoder(sparse_output=False)
#OneHotEncoder 반환값을 별도로 받아야 하는데 리턴값이 numpy 배열 형태임 
n = ohe.fit_transform(demo_df) #fit -> transform 
print( ohe.get_feature_names_out()) #필드 이름 
print(n) #입력값으로 써야 한다 

