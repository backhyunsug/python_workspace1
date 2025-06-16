#큐를 사용한다. 다음 방문위치를 큐에 넣어놓고 큐에서 빼서 확인하면서 
#탐색하는 방식이다. 동적계획법 너무 고급 
"""
6 6
011111
010001
010101
010100
000110
111110
"""
N, M = map(int, input().split()) 
arr = []
for i in range(N):
    temp = list(input())  # str-> list 
    #print(temp)
    temp = list(map(int, temp))
    arr.append(temp)

def printArray(arr, N):
    for i in range(N):
        print(arr[i])

printArray(arr, N)

visited = [ [False]*M for _ in range(N) ]

from collections import deque 
def bfs(y, x): #시작값 y:행, x:열
    dx = [-1,1,0,0 ]
    dy = [0,0,-1,1 ]
    visited[y][x] = True #방문을했음 
    #bfs는 큐를 활용할 탐색기법이다 

    q = deque()
    q.append((y,x)) #튜플로 추가하기 
    while q: #큐가 빌때까지 
        y, x = q.popleft()
        for i in range(len(dy)):
            #새로운좌표만들기; 
            ny = dy[i] + y 
            nx = dx[i] + x 
            if not(0<=nx<M and 0<=ny<N):
                continue 
            if arr[ny][nx]==1:
                continue 
            if not visited[ny][nx]:
                visited[ny][nx]=True 
                #큐에 방금 좌표값 추가 
                q.append((ny, nx))
bfs(0,0)
printArray(visited, N)        