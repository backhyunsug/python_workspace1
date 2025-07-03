"""
y = ax + b      자동으로 a 와 b 를 찾아내는 과정 
                 a, b        2  1 
                 x=[1,2,3,4,5,6,7, ....]
                 y=[...................] 기대값
                 real_y=[............... ]  오차의제곱의합 : 4843783873
                 a, b    3 1 

y = w1x1 + w2x2  + w3x3   + w4x4 ......   + b  

y 에 영향을 미치는 x1,x2,..............
         기울기들 w1,w2,.................  가중치   

모든 데이터(필드, 특성, 픽처)들의 단위가 비슷해야 한다. 
서포트벡터머신 머신러닝 , 딥러닝 = 반드시 단위가 비슷해야 한다. 
단위맞추는 과정을 정규화라고 한다. 
"""
#파일명 : exam11_4.py
#데이터표준화

import pandas as pd 

data = pd.read_csv('./data/auto-mpg.csv')
print(data.info())
print(data.head())

#컬럼명 변경하기 => 프로그램 안에서 바꾸기 
data.columns=['mpg', 'cyl', 'disp', 'power', 'weight', 'acce', 'model']
print(data.head())

#정규화( 정규화하고자하는값 - 최소값)/(최대값-최소값)
data['mpg2'] = (data['mpg']-data['mpg'].min())/(data['mpg'].max()-data['mpg'].min())

#단위환산 
mpg_unit = 1.60934/3.78541
data['kpl'] = (data['mpg']*mpg_unit).round(2) #소수점이하 2자리 반올림 
print(data.head()) 


