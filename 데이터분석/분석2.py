#파일명 : 분석2.py 
import pandas as pd 

#분석에서 데이터가 크게 두종류가 있다 - 분석방법이 다르다 
#연속형데이터, 평균값이 중요하다 연비 40, 30, 25, 회귀분석, 회귀(Regressior)    
#불연속형데이터, 범주형, 카테고리, 평균값이 중요하지 않다. 빈도수가 중요하다. 
#발생빈도수 - 실린더수 3 5 4 6 8 , 분류분석, 로지스틱회귀, 분류(Classifier)

#header가 3번째 줄에 있음 
data = pd.read_csv("./data/auto-mpg.csv")

#value_counts 각 데이터별 고유카운트-빈도수, 발생빈도수를 카운트 한다  
print( data['model-year'].value_counts())

#평균, 최대, 최소 
print("연비평균 : ", data['mpg'].mean())
print("연비최대 : ", data['mpg'].max())
print("연비최소 : ", data['mpg'].min())
print("연비중간 : ", data['mpg'].median())
print("연비분산 : ", data['mpg'].var())
print("연비표준편차 : ", data['mpg'].std())

print("1사분위수 : ", data['mpg'].quantile(0.25))
print("2사분위수 : ", data['mpg'].quantile(0.5))
print("3사분위수 : ", data['mpg'].quantile(0.75))