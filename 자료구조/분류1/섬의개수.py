#큐를 사용한다. 다음 방문위치를 큐에 넣어놓고 큐에서 빼서 확인하면서 
#탐색하는 방식이다. 동적계획법 너무 고급 
"""
5 4
1 0 1 0 0  
1 0 0 0 0
1 0 1 0 1
1 0 0 1 0

1. 2차원배열 전체를 검색해야 한다. 어디가 섬인지 모르니까 
i, j 놓고 전체 배열을 검색한다. 
  첫번째 1을 찾는다. dfs나 bfs를 작동한다. 
  visited 배열         0, 0방이 1      
  0 0 0 0 0           1 0 0 0 0       1 0 0 0 0    
  0 0 0 0 0           0 0 0 0 0       1 0 0 0 0 
  0 0 0 0 0           0 0 0 0 0       0 0 0 0 0
  0 0 0 0 0           0 0 0 0 0       0 0 0 0 0

  1 0 0 0 0           1 0 0 0 0      dfs나 bfs한번 끝나고 오면 섬찾음 1 
  1 0 0 0 0           1 0 0 0 0 
  1 0 0 0 0           1 0 0 0 0
  0 0 0 0 0           1 0 0 0 0

  두번째               0,1 
  1 1 0 0 0  
                      0,2 가 1이었음   dfs 또는 bfs       섬찾음 2 
  1 1 1 0 0 

  1 1 1 1 1 
  1 1 1 1 1
  1 1 1 0 1                                            섬찾음 3
  1 0 0 1 0








5 4
1 1 1 0 1
1 0 1 0 1
1 0 1 0 1
1 0 1 1 1

5 5
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1

"""

M = 5  #열의개수 - 너비 
N = 4  #행의개수 - 높이 
arr = [
    [1, 0, 1, 0, 0],  
    [1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0]
]

visited = [[0]*M for _ in range(N)]
#print(visited)
def printArray(arr):
    print()
    for a in arr:
        print(a)
    


printArray(visited)

#재귀호출 -> 내부 스택을 사용한다. 
def dfs(y,x):
    #방문표시 
    visited[y][x] = 1  #
    printArray(visited)
    #탐색이 가능한 좌표배열을 만든다 
    #상하좌우대각선 
    dx = [0, 0, -1, 1, -1,-1, 1, 1] #좌우 
    dy = [-1,1,  0, 0,  1,-1, 1,-1] #상하 

    #8방향으로 새로운 좌표를 산출한다. 그래서 이동 가능한지 확인한다.
    for i in range(len(dx)):
        ny = y + dy[i] 
        nx = x + dx[i] 
        #이 좌표가 벽이야 그러면 다른 좌표 확인하기 
        if ny<0 or ny>=N or nx<0 or nx>=M:
            continue  #이 좌표는 버린다. continue문은 이 다음구문은 실행을 안하고 
        #다시 for문으로 점프한다 
        #방문을 안했거나 해당 좌표가 1일 경우에 탐색을 계속한다 
        if visited[ny][nx]==0 and arr[ny][nx]==1:
            dfs(ny, nx) #새 좌표에서 또 다른 탐색을 진행한다 

    print(y,x)

def solution(arr, M, N):
    island_cnt=0 #섬의 개수 0 

    for i in range(0, N): #행, 높이 
        for j in range(0, M): #열, 너비 
            #방문 안했고 섬이어야 dfs호출하기 
            if visited[i][j]==0 and arr[i][j]==1: 
                dfs(i,j) #탐색 한번 할때마다 
                island_cnt += 1 

    return island_cnt

print(solution(arr, M, N))




"""
M, N = map(int, input().split()) 
arr = []
for i in range(N):
    temp = input().split()  # str-> list 
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
    dx = [-1,1,0,0,-1,-1,1,1 ]
    dy = [0,0,-1,1,-1,1,-1,1 ]
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
            if arr[ny][nx]==0:
                continue 
            if not visited[ny][nx]:
                visited[ny][nx]=True 
                #큐에 방금 좌표값 추가 
                q.append((ny, nx))

def island_count():

    # cnt=0
    # bfs(0,0)
    # printArray(visited, N)
    # bfs(1,2)
    # printArray(visited, N)
    cnt=0 
    for i in range(0, N):
        for j in range(0, M):
            if arr[i][j] == 1 and not visited[i][j]:
                print(f"i={i}, j={j}")
                bfs(i,j)
                printArray(visited, N)
                print()
                print()
                cnt+=1 
    
    return cnt 

print (island_count())

#printArray(visited, N)        
"""