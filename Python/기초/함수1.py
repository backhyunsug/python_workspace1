#함수란? 특정 기능끼리 묶어놓은 코드 
#아주 옛날에 프로그램을 스파게티 코드 
#결과만 나오면 좋다. => 
#모듈 : 일을 작게 나누어서 처리하자 나눈일의 단위 
#모듈은 프로시저와 함수로 나눈다.
#프로시저는 일이 끝나고 값을 반환하지 않는다 
#함수는 일이 끝나고 값을 반환한다. 
#c언어가 프로시저와 함수를 합쳐서 함수라고 부른다.
#파이썬의 경우는 def 키워드로 시작해서 
#def 함수이름(매개변수들):
#     ...........
#      return 값 원칙이 값 하나만 리턴한다. 파이썬도 값을 하나만 보낸다. 
#                만일 여러개 보내면 tuple타입으로 묶어서 전달한다. 
#장점 : 유지보수가 편해진다. 
#      반복적인 일을 처리하고자 할때 함수를 만들면 간결하게 처리할 수 있다 
#      구조적프로그래밍(c언어), 객체지향 프로그램에서 필수다
# 함수는 15줄 넘어가면 안된다. A4용지 한장 넘어가면 안된다. 
# 
# 
# 
def print_line(): #함수를 정의 한다 
    #pass  #파이썬은 함수가 되었던 아니면  for, if나 등등등 코드가 없이  if 조건식 {}
    print("="*30)
    #함수에서 반환값을 안주면 None 이 온다 

print_line() #함수를 호출한다. 
print_line()
print_line()
print_line()
print_line()
print_line()

print( print_line() )

#1부터 N까지의 합계를 구하는 함수 만들기 
def sigma(limit): #작은 프로그램 단위 입출력
    #limit : 매개변수, 매개체를 말한다. 함수 외부에서 함수 내부로 값을 전달하기 위한 목적
    s = 0
    for i in range(1, limit+1):
        s += i 
    return s 

print( sigma(10) )
print( sigma(100) )
print( sigma(1000) )
print( sigma(10000) )

"""
과제1. 정수를 받아가서 짝수이면 True 짝수가 아니면 False를 반환하는 함수 

과제2. 윤년 4 년마다 윤년 
          100년에 윤년이 아님 
          400년에 한번씩 윤년이다. 
          함수를 만들어서 연도를 주면 윤년일 경우 True 윤년이 아니면 False를 반환해라 
 
"""
def isEven(n):
    if n%2==0:
        return True 
    return False 

def isLeap(year):
    if (year%4==0 and year%100!=0) or year%400==0:
        return True 
    return False 

for i in range(1, 11):
    print(f"{i} = {isEven(i)} ")  

print("윤년확인")
for i in range(2000, 2026):
    print(f"{i} = {isLeap(i)}")


