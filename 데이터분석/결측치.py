import numpy as np 
import pandas as pd 


s1 = pd.Series([1,2,3,4,np.nan]) #프로그램에서 nan값을 직접 입력할때는 np.nan을 사용해야 한다 
print(s1.isnull()) 
print(s1.isnull().sum()) #[ False, False, False, False, True]
#c언어 True는 1이고 False는 0으로 해석  
#[0,0,0,0,1].sum() 

data = pd.read_csv("./data/data.csv")
print(data.shape) 
print(data.info())
print(data.describe())

print( data['height'].value_counts()) 
print( data['height'].value_counts(dropna=False)) 

print( data['height'].isnull().sum()) 
print( data['weight'].isnull().sum()) 

#결측치가 발생하면 행이나 열을 삭제시키거나 평균값이나 중간값으로 대체를 한다. 
#표준편차가 크면 평균값 보다는 중간값으로 대체를 하는것이 좋다 

data2 = data.dropna( how='any', axis=0) #행중에 NaN값 있으면 행 전체를 삭제해라 
print(data2.shape)#25, 3으로 준다 

data2 = data.dropna( how='any', axis=1) #열중에 NaN값이 하나라도 있으면 삭제한다. 
print(data2.shape)#30, 1으로 준다 name열 빼고 전체가 삭제된다. 

#thresh 라는 옵션 - 최소한의 실효성있는 데이터 개수를 유지해라
data2 = data.dropna(thresh=28, axis=1) #데이터개수가 27개인 열은 삭제된다. 
print(data2.shape)
print(data2.head())

data = pd.read_csv("./data/auto-mpg.csv")
print( data.info() )

print(data['horsepower'].isnull().sum())
data = data.dropna(how='any', axis=0)
print(data.shape)









