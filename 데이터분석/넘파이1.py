import numpy as np 
#numpy - 분석용라이브러리, c언어 

a = [1,2,3,4,5]
b = [2,4,6,8,10]
c = a+b 
print(c)

#머신러닝, 딥러닝 둘다 취급하는 데이터 타입이 ndarray 타입이다. 
a1 = np.array(a)  #타입을 list -> ndarray 타입으로(C언어의 배열, 속도가 엄청 빠름)
print(a1, type(a1))
b1 = np.array(b)
c1 = a1 + b1      #수학의 벡터연산을 수행한다. 
                  #스칼라연산 - 요소 하나씩 더하기 
                  #벡터연산 - 벡터 통째로(배열 통째로 연산을 수행함, forX)
print(c1)   

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = 2*x + 3    #수학적 연산에 집중된 방식, 벡터연산을 수행한다. 
print( y )

#기초통계
print("평균", np.mean(x))
print("중위값", np.median(x))
print("최소값", np.min(x))
print("최대값", np.max(x))
print("표준편차", np.std(x))
print("분산",    np.var(x))





