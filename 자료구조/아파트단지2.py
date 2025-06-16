"""
7
0110100
0110101
1110101
0000111
0100000
0111110
0111000
"""

N = 7 #높이
M = 7 #너비 
arr = [ 
    [0,1,1,0,1,0,0],
    [0,1,1,0,1,0,1],
    [1,1,1,0,1,0,1],
    [0,0,0,0,1,1,1], 
    [0,1,0,0,0,0,0],
    [0,1,1,1,1,1,0],
    [0,1,1,1,0,0,0]
]

visited = [ [0]*M for _ in range(N)] 

def printArray(arr):
    print()
    for a in arr:
        print(a)

from collections import deque #큐라이브러리 

#큐를 이용한 탐색
def bfs(y, x):
    dx = [0, 0, -1, 1 ] #좌우 
    dy = [-1,1,  0, 0 ] #상하 
    
    visited[y][x]=1 #방문
    q = deque() 
    #1.큐에 방문값을 입력한다 ]
    q.append((y,x))

    cnt=1
    while q: #큐에 탐색할것이 남아 있는 동안 탐색을 계속한다 
        #큐에서 좌표값을 하나 꺼내온다 
        cy, cx = q.popleft() #큐에 넣을때 y값부터 넣었으므로 ,y부터 가져온다
        for i in range(len(dx)):
            ny = cy + dy[i] #새좌표를 만든다 
            nx = cx + dx[i] 
            #벽인지 확인하기 
            if ny<0 or ny>=N or nx<0 or nx>=M:
                continue 
            if visited[ny][nx]==0 and arr[ny][nx]==1:
                #몇번만에 찾았는지, 이동한 자릿수마다 카운트 증가  
                visited[ny][nx] = visited[cy][cx]+1
                cnt+=1
                q.append((ny, nx)) 
    
    return cnt
    
def solution(arr, N):
    cntList = []
    for i in range(N):
        for j in range(M):
            if visited[i][j]==0 and arr[i][j]==1:
                cnt = bfs(i,j) #탐색하기
                cntList.append(cnt)
    return cntList 
                #print(cnt)

print( solution(arr, N) )




