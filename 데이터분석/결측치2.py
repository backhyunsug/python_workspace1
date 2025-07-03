import pandas as pd 

#header가 3번째 줄에 있음 
data = pd.read_csv("./data/data.csv")

print(data.head())
print(data.info())

print("height 결측치 : ", data["height"].isnull().sum())
print("weight 결측치 : ", data["weight"].isnull().sum())

#데이터프레임 전체의 결측치를 확인하기 
print(data.isnull().sum())

mean_height = data['height'].mean() 
mean_weight = data['weight'].mean() 

#fillna(대체값, inplace=True) inplace=True이면 원본데이터를 바꾼다. 그게 아니면 반드시 반환을 받아야 한다 
data['height'].fillna(mean_height, inplace=True) 
data['weight'].fillna(mean_height, inplace=True)
 
print("누락데이터 교체 후")
print(data.isnull().sum())