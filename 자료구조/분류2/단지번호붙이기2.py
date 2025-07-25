"""
마을의 위성사진을 본 철수는 평지와 호수로 나뉘어져 있다는 것을 알았다.

이 사진을 통해 호수가 몇 개가 있는지 파악하려고 한다.

상, 하, 좌, 우, 대각선으로 연결되어 있으면 하나의 호수로 간주한다면
철수의 마을에 몇 개의 호수가 있는지 파악할 수 있는 프로그램을 작성하자

첫째 줄에는 마을의 크기 N이 주어진다. (4<=N<=100)
둘째 줄부터 N줄까지 마을 정보가 공백 없이 주어진다.
(0은 평지 1은 호수임)

5
01010       
10001
01010
00100
10000

N x N의 입력 배열 생성

방문 여부를 저장할 visited 배열 생성

배열을 전체 순회하면서

아직 방문하지 않은 1을 만나면 DFS 탐색 시작

DFS 탐색을 통해 하나의 호수에 포함된 모든 1을 방문 처리

DFS가 끝나면 호수 1개 완료 → 카운트 증가

총 DFS 실행 횟수가 곧 호수의 개수
"""

import sys
sys.setrecursionlimit(10000)  # DFS 깊이 제한 해제

# n = int(input())
# graph = [list(map(int, list(input().strip()))) for _ in range(n)]

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
    
print1(graph)
visited = [[False] * n for _ in range(n)]

# 8방향
dx = [-1, 1, 0, 0, -1, -1, 1, 1]
dy = [0, 0, -1, 1, -1, 1, -1, 1]


#시스템 내부의 스택을 활용해서 그래프 탐색을 한다. 그래서 깊이 우선 탐색임 
def dfs(x, y): 
    visited[x][y] = True  #방문, 중요 방문을 해야 함수가 리턴한다  

    cnt=1
    for d in range(8): #8방향으로 갈 수 있는지 확인을 한다 
        nx = x + dx[d] #개로운 좌표값을 구한다 
        ny = y + dy[d]

        if 0 <= nx < n and 0 <= ny < n: #벽을 벗어나지 말아야 한다 
            if graph[nx][ny] == 1 and not visited[nx][ny]:
                cnt+=dfs(nx, ny)#재귀호출
    return cnt 


lake_count=[]
for i in range(n):
    for j in range(n):
        if graph[i][j] == 1 and not visited[i][j]:
            cnt = dfs(i, j) #시작위치 선정 
            lake_count.append( cnt )

print(lake_count)