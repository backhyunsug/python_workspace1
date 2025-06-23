n = int(input())

memo = [0] * (n+ 1)

def fibo(n):
    # 종료 조건 추가하기
    if n < 2:
        return n

    # 메모이제이션1 : 저장값 반환
    if memo[n]:
        return memo[n]

    # 메모이제이션2 : 새로운 값 저장
    memo[n] = fibo(n-1) + fibo(n-2)
    return memo[n]

print(fibo(n))