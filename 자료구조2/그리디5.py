"""
큰 수의 법칙 (중상 난이도)
문제 설명:
N개의 수로 이루어진 배열이 있습니다. 
이 중 M번 더하여 가장 큰 수를 만들되, 
특정 숫자는 연속해서 K번까지만 더할 수 있습니다.

N = 5, M = 8, K = 3  
배열 = [2, 4, 5, 4, 6]
6 + 6 + 6 + 5+
6 + 6 + 6 + 5 = 
"""
def big_number_rule(N, M, K, numbers):
    numbers.sort()
    first = numbers[-1]
    second = numbers[-2]
    
    full_patterns = M // (K + 1)
    remainder = M % (K + 1)
    
    result = (full_patterns * (first * K + second)) + (remainder * first)
    return result

print(big_number_rule(5, 8, 3, [2, 4, 5, 4, 6]))  # 출력: 46