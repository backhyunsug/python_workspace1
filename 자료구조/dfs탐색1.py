#https://wikidocs.net/206317
"""
6 6
0 1 1 1 1 1
0 1 0 0 0 1
0 1 0 1 0 1
0 1 0 1 0 0 
0 0 0 1 1 0
1 1 1 1 1 0

str-> list -> map(int, ) -> list
6 6
011111   
010001
010101
010100 
000110
111110
"""

#map은 list 의 요소에 앞에서 전달해준 수식 또는 함수를 적용한다 
"""
data = input().split()
n = int(data[0])
m = int(data[1])
print(n, m)
"""
N, M = map(int, input().split()) #문장으로 받는다. 문장을 잘라줘서 단어로 만든다
#print(N, M)
arr = []
for i in range(N): 
    #map함수는 iterable 아직 진행이 안되서 
    #filter, range, zip 등등들은 for문 안에서 호출하거나 list로 감싸줘야 작동을 한다
    temp = map(int, input().split())
    arr.append( list(temp))  

def printArray(arr, N):
    for i in range(0, N):
        print( arr[i] )

printArray(arr, N)

"""
       0  1  2  3  4 
  --------------
  0 |  0                    둘러싸고 있는 공간  상하좌우4 대각선4개방향         
  1 |  0                     1, 0  0,0  2,0 
  2 |  0                    방문테이블 => 내가 갔던 위치를 1로 바꿔쳐서 다 표시를 한다 

"""
#방문테이블  파이썬 미리 메모릴 할당 1차원 []*개수
visited = [[False] * M for _ in range(N)]

visited=[] 
for i in range(N):
    visited.append( [False]*M)
printArray(visited, N)