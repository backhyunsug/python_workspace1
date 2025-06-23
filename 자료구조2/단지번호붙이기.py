"""
https://www.acmicpc.net/problem/2667

7
0110100
0110101
1110101
0000111
0100000
0111110
0111000


큐를 이용하여 풀기 
"""
from collections import deque

n = int(input())
graph = []
for i in range(n):
    data = list(input().strip())
    #print(data)
    data = map(int,data)
    graph.append(list(data))

def print1(data):
    for i in range(n):
        print(data[i]) 
    print()
    
#print1(graph)
#visited = [[False] * n for _ in range(n)]


# 8방향 (상, 하, 좌, 우)
dx = [-1, 1, 0, 0 ]
dy = [0, 0, -1, 1 ]

def bfs(x, y, grid, visited, n):
    queue = deque()
    queue.append((x, y)) #큐에 시작위치를 입력한다 
    visited[x][y] = True #방문한 흔적을 저장한다 

    count=1
    while queue: #큐가 빌때까지 작업한다 
        cx, cy = queue.popleft() #큐로부터 첫번째 좌표값을 꺼낸다 
        for d in range(4): 
            nx = cx + dx[d]#새로운 좌표를 찾는다 
            ny = cy + dy[d]
            if 0 <= nx < n and 0 <= ny < n:  #벽을 벗어나지 않고 값이 1일 경우에 그리고 방문하지 않았을 경우에
                if not visited[nx][ny] and grid[nx][ny] == 1:
                    visited[nx][ny] = True #방문했음을  표시하고 
                    queue.append((nx, ny)) #큐에 다시 넣는다.
                    count+=1      
    return count

def count_lakes(grid):
    n = len(grid)
    visited = [[False]*n for _ in range(n)]
    count = 0

    cnt=[]
    for i in range(n):
        for j in range(n):
            if not visited[i][j] and grid[i][j] == 1:
                cnt.append(bfs(i, j, grid, visited, n))
                count+=1
    return count, cnt


print( count_lakes(graph))
