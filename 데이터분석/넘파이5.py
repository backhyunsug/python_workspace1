#argmax, argmin 
import numpy as np 

#랜덤값은 컴퓨터의 내부의 시계를 이용해서 추출하는데 
#보고서를 쓸때 값이 계속 바뀌면 문제가 되는 경우가 있다. 
#고정시키고싶을때 seed 값 
np.random.seed(1234) #0~아무값이나 주면 된다. 
a = np.random.rand( 5 )
print(a)
print(np.max(a), np.argmax(a)) #argmax 큰값이 있는 위치값을 반환한다

a = np.random.rand( 5 )
print(a)
print(np.max(a), np.argmax(a)) #argmax 큰값이 있는 위치값을 반환한다

#문제1. 가우스분포에 따른 랜덤값을 5개씩 10개 생성해서 각 행마다 
#젤큰값하고 큰값위치 찾아서 출력하기 
a = np.random.rand(50)
a = a.reshape(10,5)
for i in range(0, 10):
    print( np.argmax(a[i]), np.max(a[i]) ) 