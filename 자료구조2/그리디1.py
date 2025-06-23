"""
그리디 - 욕심쟁이, 탐욕 
반드시 정렬을 해야 문제를 풀 수 있다 

거스름돈으로 줄 수 있는 동전이 [500원, 100원, 50원, 10원]일 때,
거스름돈 금액 N원을 입력받아 동전의 최소 개수를 구하라.

"""

def get_change(n):
    coins = [500, 100, 50, 10]
    count = 0
    for coin in coins:
        count += n // coin
        n %= coin
    return count

# 실행
print(get_change(1260))  # 출력: 6
