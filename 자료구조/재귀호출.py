"""
재귀호출 : 자기가 자기를 호출하는 함수 
         
A             B 
  callB()     callA()

의도치 않은 재귀호출  안끝난다. 서로 계속 호출을 반복하면 스택에 
계속 쌓여서 스택 overflow가 발생한다 
무한대로 호출하지 않게 만들어줘야 한다 

장점 : 코드를 간결하게, 트리순회시 재귀호출이 유용, 퀵정렬 알고리즘 
단점 : 메모리를 많이 사용하고 속도 느림 
"""
#반드시 언젠가는 끝내야 해서, 매개변수를 갖고 다니면서 이값을 증가 또는 
#감소를 시켜서 끝나는 상황을 만들어줘야 한다. 
#1~10까지 출력을 하려고 한다
# 1 , 2, 3,  
def recursive_call(n):
    if n > 10: #함수의 종료조건이 중요하다 
        return   
    print(n)
    recursive_call(n+1) 

recursive_call(1)           
#1   recursive_call(2)
#1 2 recursive_call(3)
#1 2 3 recursive_call(4)
#1 2 3 4 recursive_call(5)
# ........ 
# 
def recursive_call2(n):
    if n<1:
        return
    print(n)    
    recursive_call2(n-1)

recursive_call2(10)

#피보나치수열 
# f(1) = 1
# f(2) = 1
# f(3) = f(1) + f(2) = 2
# f(4) = f(2) + f(3) = 3
# f(5) = f(3) + f(4) = 5 
# f(n) = f(n-2) + f(n-1)  = 8
def fibonacci(n):
    if n==1 or n==2:
        return 1
    return fibonacci(n-2) + fibonacci(n-1) 

print("3번째요소 ",  fibonacci(3) )
print("6번째요소 ",  fibonacci(6) )

#1~n 까지의합계를구하는걸 재귀호출로 작성하기
# 1 + 2 + 3 + .......... (n-1) = f(n)
# f(n) = n + f(n-1)
# f(1) = 1 
# f(2) = f(1) + 2
# f(3) = f(2) + 3
# f(n) = f(n-1) + n

def mysum(n):
    if n<=1:
        return 1
    return mysum(n-1) + n

print( mysum(10))
