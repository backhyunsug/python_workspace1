import pandas as pd 

df = pd.read_csv("./data/auto-mpg.csv")
#print(df)

print(df.head())
print(df.tail())

row, col = df.shape 
print("데이터전체개수 ", row, col)

#info ->데이터 정보를 요약해서 보여준다 
print( df.info()) 

#통계량 - 기초통계학 중요 : 평균, 개수, 표준편차(std),최소값,최대값, 중간값(2/4), 1/4분위수
print(df.describe())


print( df["cylinders"]==8)

#df2 = df[조건식] 조건식이 True 데이터 셋만 가져온다 
df2 = df[ df["cylinders"]==8]
print(df.shape)
print(df2.shape)
#value_counts() - 카테고리형 데이터  2 4 8 
print(df2["cylinders"].value_counts())

#실린더-카테고리타입 , 평균이나 표준값 따위가 의미가 없는 데이터 타입 
#value_counts() - 분할표 
print(df["cylinders"].value_counts())

#모델발표연도가 70이고 연비가 25이상(and, or-python연산자는 사용못해)
import numpy as np   
df2 = df[ np.logical_and( df["model-year"]==70 , df["mpg"]>=25) ] 
print(df2)

#모델발표연도가 70이고 연비가 25이상(and, or-python연산자는 사용못해) 데이터 건수 
print(df2.shape[0])

#모델과 모델별 제품 개수 
print(df["model-year"].value_counts())
#문제1. 80년대에 나온 제품들을 모델과 개수로 나타내기 
print(df[df['model-year']//10==8].value_counts())
print(df[np.logical_and(df['model-year']>=80, df['model-year']<=89)].value_counts())
#문제2. 73, 78, 76년에 나온 제품만 모두 출력하기 
"""
df2 = df[ (df['model-year']==73) | (df['model-year']==78) | (df['model-year']==76) ]
print(df2)
df2 = df[ (df['model-year']>=80) & (df['model-year']<=89) & (df['mpg']>=25) ]
df[(df['model-year'].isin(range(80, 90))) & (df['mpg'] >= 25)]
"""
#문제3. 80년대에 출시한 제품중에서 연비가 25 이상인 제품의 정보만 출력하기 
print(df[np.logical_and(df['model-year']//10==8, df.mpg>=25)])
#문제4. 70년대에 출시된 모델중에서 실린더가 4개인 제품의 정보만 출력하기
#  
#문제5. 80년대에 출시된 모델들을 실린더 개수와 제품개수로 출력하기 

# | or 비트연산자   & and 연산자
#df2 = df[ (df['model-year']==73) | (df['model-year']==78) | (df['model-year']==76) ]
#

a = 5 & 7  #  0101 & 0111 = 0101 
print(a)
