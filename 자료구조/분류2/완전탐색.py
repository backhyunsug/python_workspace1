"""
완전탐색 - 모든 경우의 수 

2개를 조합하기  : 2중 for 
3개를 조합하기  : 3중 for 
4개를 조합하기  : 4중 for 
5개를 조합하기  : 5중 for 
............  코드를 전체를 고쳐야 한다 
............  재귀호출을 사용한다 

A B
0 0        A A
0 1        A B
1 0        B A 
1 1        B B

A A A    8가지  
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
            dfs   dfs, dfs, 
A A A A     ['A', 'A', 'A', 'A'] -> 출력    
            dfs 
A A A B     ['A', 'A', 'A', 'B'] -> 출력 
            dfs, dfs 
            ['A', 'A', 'B', 'A'] -> 출력 
            dfs 
            ['A', 'A', 'B', 'B'] -> 출력
                        
            배열, depth(깊이) 1,2,3,4, 전체배열길이:length     
"""

def dfs(arr, depth, length):
     #이 함수가 끝나는 요건 
     if depth==length:
          print(arr) #배열의 내용출력을 하고 
          return #함수 종료하기 
     #처음에 depth는 0 부터 시작해서 1 2 3 4
     arr[depth]='A'
     dfs(arr, depth+1, length)

     arr[depth]='B'
     dfs(arr, depth+1, length)

length = 6     
arr = [""]*length 
dfs(arr, 0, length) 


#파이썬은 알고리즘을 제공하고 있음 
from itertools import combinations
from itertools import permutations
data = [1,2,3,4]
print(combinations(data,3)) #iterator 를 내보내기위한 준비 
result = list(combinations(data,3))
print(result) 
result = list(permutations(data, 4))
print(result)