import numpy as np 

x = np.arange(20)
print(x)
print(x[:10]) #0~9번방까지 
print(x[10:]) #10번방~끝까지
print(x[::-1]) #역순으로 
print(x[10:2:-1])
print(x[10:0:-2])
print(x[1:3])
print(x[2:7])

#조건식 
print ( x>=10) #[False, False, ... True, True]

#파이썬의 리스트는 조건식이 적용되지 않는다. 
#a = [1,2,3,4,5]
#print( a>=3)

print( x[ [1,3,5,7,9] ]    )
print( x[ x>=10 ]    )

#x값이 짝수의 경우만 
print( x[x%2==0] )

#3의 배수이면서 5의 배수인것을 추출하고 싶다 
#x%3==0 and x%5==0 - and 연산은 numpy연산자가 아니라서 문제가 된다. 

print( x[ np.logical_and(x%3==0,  x%5==0)] )

x = np.array([1,2,3,4,5])
print(x[ [True, True, True, False, False ]])


#2차원의 경우에 slicing 
k = np.array([ [1,2,3,4,5],
      [6,7,8,9,10],
      [11,12,13,14,15],
      [16,17,18,19,20]])

k2 = np.arange(1, 21)
k2 = k2.reshape(4,5)
print(k)
print(k2)

print(k[:]) #전부다 

print(k[:1])  #1행만 
print(k[:2])  #2행까지 
print(k[:3])  #3행까지 

print("###############################")
print(k[::2])
print(k[:2, :2])  #numpy만 된다 
print(k[2:4, 3:5])

#numpy배열 합하기 
a = np.array([1,2,3,4,5])
b = np.arange(6, 11)
c = np.concatenate( (a, b, np.arange(11,21)) ) #tuple로 받아간다 
print(c)

c1 = np.array_split(c, 3) #ndarray 를 쪼개서 새로운 배열로 만든다
#print(c)
print(c1)

#검색은 where 를 사용한다 
print( np.where( a%2==0))

#정렬 
c1 = np.random.randint(1, 100, 10)#1~100까지 랜덤값 10개 추출하기 
print(c1)
c2 = np.sort(c1) #정렬된 데이터를 반환한다  
print(c2)  





















