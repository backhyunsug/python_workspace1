#튜플은 다른언어에 아직 없는 타입이었는데 최근에 추가되고 있다 
#튜플 -> 우리가 점심때 플리마켓에 놀러갔다. 이것저것 사다가 손에 다 못들면 비닐봉지 
# 하나 얻어서 비닐봉지에 담는다. 이 때 비닐봉지와 같은 역할을 하는것이 튜플이다. 
# 괄호를 사용한다. 인덱싱과 슬라이싱을 지원한다.
# 인덱싱이나 슬라이싱을 통한 업데이트는 안된다.
# read only list 형, 속도가 빠르다 
# "이름 %s 나이 %d " % ("홍길동", 45)  괄호가 tuple타입이다 
a = (1,2,3,4,5)
print( a, type(a))

a = 5
b = 7 
c = 9 

a,b,c = 5,7,9 #tuple활용 
print(a,b,c)

#함수-> 코드를 묶어놓은것, 함수는 값을 하나만 반환해야 한다. 
#파이썬의 경우에는 우리가 여러개를 반환하면 이걸 알아서 tuple로 묶어서 하나를 보낸다.  
def myfunc1():
    return 3,4
a = myfunc1() 
print( type (a) )

b,c = a 
print(b,c)

a = 5 
b = 7

#두개의 변수값을 서로 exchange, swap
c = a 
a = b 
b = c 
print( "A=",a, "B=",b)

b, a = a, b   #tuple를 써서 값을 swap 할 수 있다
print( "A=",a, "B=",b)

#튜플은 read only이다. 삭제도 업데이트도 안된다. 
a = (1,2,3,4,5,6,7,8,9,10)
print( a[0] )
print( a[1] )
print( a[2] )

#del a[2] #삭제안됨 
#a[0] = 11
 
print( a[:5])
print( a[2:5])
print( a[::-1])

b = a + a
print(b)

b = a * 3
print(b)

