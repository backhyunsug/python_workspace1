#1~10까지의 합계 구하기 
#변수 - 1~10숫자 세는 변수
#더해지는 값 - 누적값을 저장할 변수가 필요하다. 누적이 된다. 
#for문 밖에서 0으로 값이 초기화 되어야한다
# 0 +1 
# 0 + 1 + 2
# 0 + 1 + 2 + 3
sum = 0   # sum = sum + i 
limit = int(input("limit : "))
for i in range(1, limit+1):
    sum = sum + i 
    print(f"i={i} sum={sum}") 

#문제1. 정수를 5개 입력받아서 합계를 구하라 
#입력받고 더하고 입력받고 더하고 
sum = 0 
for i in range(1, 6):
    n = input("숫자 : ")
    sum = sum + int(n) 
print ( sum )

#숫자를 10개 입력받아서 각각 짝수의 홀수의 합과 평균을 구하는 프로그램을 작성하기 

