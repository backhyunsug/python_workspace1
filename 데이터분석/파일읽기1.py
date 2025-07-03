import pandas as pd 
df = pd.read_csv("./data/score.csv") 
#데이터프레임으로 만들면 자동으로 행 인덱스가 부여된다.
print(df.head()) 
print("컬럼명 : ", df.columns)
print("인덱스 : ", df.index)

df["total"] = df.kor + df.eng + df.mat  #키값에 하이픈이 들어갈 경우 이렇게 쓸 수 없다
df["avg"] = df.total//3                 #키추가시에는 df["키"] 형태로 써야 한다  
print(df )


