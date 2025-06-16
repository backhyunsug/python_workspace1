#전체를 살펴봐야 한다
"""
[[1, 2], [2, 3], [2, 1]]	4	4	2
[[1, 2], [2, 3], [2, 1]]	1	7	0
[[3, 3], [3, 3]]	        7	1	6
[[3, 3], [3, 3]]	        6	1	-1

1. 둘다 잡히면 안된다
2. A의 이익이 최대가 되도록 한다

A의 훔친 총합에는 제한이 없음. 다만, 최소값을 구하는 것이 목표.
B가 훔쳐야 하는 목표 총합. 이 이상이 되면 B의 목표는 달성한 것. B가 최대한 많이 훔쳐야 한다

float('inf') 양의 무한대 
n = 10   # A는 최대 10까지 허용 (넘으면 의미 없으니 중단)
m = 5  

info = [[3, 5], [4, 2]]   
모든 경우의 수를 따져야 한다 

인덱스	선택	A합	B합
0 → A, 1 → A	[3, 4]	7	0
0 → A, 1 → B	[3, 2]	3	2
0 → B, 1 → A	[5, 4]	4	5  B 조건 만족
0 → B, 1 → B	[5, 2]	0	7  B 조건 만족

A  A
A  B
B  A
B  B


A A A 
A A B
A B A
A B B
B A A 
B A B
B B A
B B B 

A A A A                    
A A A B
A A B A                    
A A B B
A B A A
A B A B
A B B A
A B B B

B A A A
B A A B
B A B A 
B A B B
B B A A
B B A B
B B B A
B B B B




A  - A  - A  -A
             -B 
        - B  -A
             -B 
   - B  - A  -A
             -B
        - B 
B  
"""
def dfs(arr, depth, length):
    if depth == length:  #길이가 차면 arr을 문자열로 바꾸어서 리턴한다 
        print(arr) #계속 배열이 쌓이다가 맨 마지막에 출력하고 리턴한다 
        return

    arr[depth] = 'A'
    dfs(arr, depth + 1, length)

    arr[depth] = 'B'
    dfs(arr, depth + 1, length)

def print_combinations(length):
    arr = [''] * length  # 고정된 길이의 문자 배열
    dfs(arr, 0, length)
#  dfs(arr['A'], 0, 4)
#  dfs(arr['B'], 0, 4)
# # 예: 길이 3짜리 조합 출력
print_combinations(4)

# #재귀호출을 이용한 탐색 방법 

# def solution(info, n, m):
#     global answer

#     answer = n
#     visited = set()

#     def dfs(i, a, b):
#         global answer

#         visited.add((i, a, b))
#         if a >= n or b >= m: return 
#         if a >= answer: return
#         if i == len(info) and a < answer:
#             answer = a
#             return

#         if (i+1, a, b+info[i][1]) not in visited:
#             dfs(i+1, a, b+info[i][1])
#         if (i+1, a+info[i][0], b) not in visited:
#             dfs(i+1, a+info[i][0], b)

#     dfs(0, 0, 0)

#     return answer if answer != n else -1


def solution(info, n, m):
    # 각 항목에 (a, b, a/b 비율) 추가하고 정렬
    rate = sorted(
        [[a, b, a / b] for a, b in info],
        key=lambda x: (-x[2], -x[1])  # a/b 내림차순, b 내림차순
    )
    visited = [False] * len(rate)

    a = 0
    b = 0

    # B가 훔치는 루프
    for i in range(len(rate)):
        item = rate[i]
        if m > b + item[1]:
            b += item[1]
            visited[i] = True

    # A가 훔치는 루프
    for i in range(len(rate)):
        if visited[i]:
            continue
        item = rate[i]
        if n > a + item[0]:
            a += item[0]
            visited[i] = True

    # 모든 항목이 사용되지 않았다면 실패
    if not all(visited):
        return -1
    return a

print(solution([[1, 2], [2, 3], [2, 1]],4,4))
print(solution([[1, 2], [2, 3], [2, 1]],	1,	7	))
print(solution([[3, 3], [3, 3]],7,	1))
print(solution([[3, 3], [3, 3]],6,	1))


