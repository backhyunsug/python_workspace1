num = 1000000
prime = [True] * (num + 1) # 처음엔 모두 소수로 초기화합니다.

prime[0], prime[1] = False, False # 2부터 시작하기 때문에 0, 1은 소수가 아님
for i in range(2, num + 1): # 2부터 백만까지 반복
    if prime[i]==True: # i 번째 요소가 소수라면 
        for j in range(i * 2, num+1, i): # i * 2부터 i의 배수를 순회.
            prime[j] = False # i의 배수는 모두 소수가 아닙니다.
        print(prime[:30])    

"""
            1  2  3 False 5 False 7 8 9 10
            11 12 13 
"""