
"""
6 4
0 -1 0 0 0 0
-1 0 0 0 0 0
0 0 0 0 0 0
0 0 0 0 0 1

"""
from collections import deque
M, N = map(int, input().split())

arr = []
queue = deque()
for i in range(N):
    row = list(map(int, input().split()))
    arr.append(row)
    for j in range(M):
        if row[j] == 1:
            queue.append((i, j))

dx = (0, 0, 1, -1)
dy = (-1, 1, 0, 0)
def bfs():  
    while queue:
        y, x = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= ny < N and 0 <= nx < M and arr[ny][nx] == 0:
                arr[ny][nx] = arr[y][x] + 1
                queue.append((ny, nx))

bfs()

def get_day():
    ans = 0
    for row in arr:
        if 0 in row:
            return -1

        row_max = max(row)    
        if ans < row_max:
            ans = row_max
    return ans - 1

print(get_day())