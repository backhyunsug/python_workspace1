#머신러닝, 딥러닝 - 데이터 생산 능력 

import pandas as pd 
import numpy as np 

d = pd.date_range(start="2025-07-04", end="2025-09-23")
print(d)


#period 필드와 freq 둘이 결합해서   일, 시간,,  
d = pd.date_range(start="2025-07-04", periods=30, freq='T') #periods =기간 , freq='D'
print(d)

d = pd.date_range(start="2025-07-04", periods=30, freq='W') #periods =기간 , freq='D'
print(d)

#가짜데이터 생성방법 
#np.random.randn 함수는 가우스분포를 따르는 값을 랜덤하게 생성한다. 
#가우스분포를 따르는 실수값이 중요한 이유 ]
#통계학자 => 자연계에서 얻어지는 모든값을 분석을 했음 
#대부분의 경우 양극단으로 갈수록 작아지고 중간값으로 갈수록 커지는 종모양 차트가 만들어지더라 => 정규분포 
np.random.seed(0) #랜덤값이 일정하게 시드가 같으면 같은 랜덤값이 나온다. 0~ 2의 16승 -1 이전에 컴퓨터들은 정수가 2byte였음 
ts = pd.Series( np.random.randn(12), 
               index=pd.date_range(start="2025-01-01", periods=12, freq="ME"))
#12개의 가짜 데이터를 만들었음 
#날짜가 인덱스임 

#resample 연산
ts = pd.Series(np.random.randn(100), index=pd.date_range("2025-07-04", periods=100, freq="D"))
print(ts.tail(20))  #마지막 20개만 출력

print(ts.resample('W').mean()) #D(일일) -> W(주)형식으로 전환
