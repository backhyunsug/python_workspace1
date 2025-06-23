"""
문제1) m를 km와 m로 전환하기 
2300 미터는 2km 와 300미터입니다. 

2300 1000으로 나눠서 몫을 구하면 2 
2300 - (2*1000)
2300 % 1000 
"""

# meter = int(input("미터는? "))
# km = meter //1000 
# m = meter % 1000
# #숫자 문자를 섞어서 출려할때 +말고 포맷을 활용한다.
# #fstring , python3.6부터 추가
# #f 를 붙이고 {변수명 또는 수식}
# print(f"{meter} 는 {km}km 와 {m}meter 입니다")

# 문제2) 사다리꼴이 면적 구하기  
# 사다리꼴 면적 : (윗변 + 아랫변) * 높이 /2   
# 입력 : 윗변, 아랫변, 높이 
# width = int(input('윗변' ))
# width2 = int(input('아랫변' ))
# height = int(input('높이' ))

# surface = (width + width2)* height/2 
# print(f"사다리꼴의 면적은 {surface} 입니다")



# 문제3) 철수가 식료품점에 가서 과일을 샀다 사과와 배를 샀는데 사과는 
#       한개에 5000 원이고 배는 10000원이다. 각각 사과와 배의 개수를 
#       입력받아 총금액을 구하는 프로그램을 작성하시오
# apple_count = int(input("사과 개수 : "))
# pear_count = int(input("배 개수 : "))
# total_money = apple_count*5000 + pear_count*10000 
# print(f"총 금액은 {total_money} 입니다")

# 문제4) 거스름돈 계산하기 - 10만원 짜리를 넣고 거스름돈 받기 
#       물건값이 총 : 27560 
#       거스름돈 : 72440   
#       50000 -- 1 장 
#       10000 -- 2 장 
#        5000 -- 0 장
#        1000 -- 2 장
#         500 -- 0 개 
#         100 -- 4 개
#          50 -- 0 개
#          10 -- 4 개
#입력 : 사용한 돈 
#출력 : 50000,10000, 5000, .....
#중간계산 : 거스름돈 

use_money = int(input("사용한 돈"))
change_money = 100000 - use_money
print("거스름돈 ", change_money)

temp = change_money
m50000 = temp // 50000 
temp = temp % 50000  #temporary(임시)
m10000 = temp //10000
temp = temp % 10000 
m5000 = temp //5000
temp = temp % 5000 
m1000 = temp //1000
temp = temp % 1000 
m500 = temp //500
temp = temp % 500 
m100 = temp //100
temp = temp % 100 
m50 = temp //50
temp = temp % 50 
m10 = temp //10

print("50000 ==> ", m50000, "장")
print("10000 ==> ", m10000, "장")
print(" 5000 ==> ", m5000, "장")
print(" 1000 ==> ", m1000, "장")
print("  500 ==> ", m500,  "개")
print("  100 ==> ", m100,  "개")
print("   50 ==> ", m50,   "개")
print("   10 ==> ", m10,   "개")


