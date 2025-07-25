"""
2부터 N까지 모든 정수를 적는다.
아직 지우지 않은 수 중 가장 작은 수를 찾는다. 이것을 P라고 하고, 이 수는 소수이다.
P를 지우고, 아직 지우지 않은 P의 배수를 크기 순서대로 지운다.
아직 모든 수를 지우지 않았다면, 다시 2번 단계로 간다.
"""

N = 10
K = 7
prime =[ x for x in range(0, N+1)]
prime[1]=0
P=2

def isComplete(prime):
    for i in range(0, len(prime)):
        if prime[i]!=0:
            return False
    return True
 
#1 2 3 4 5 6 7 
#0 0   0   0
P=2
flag = False
result=0
cnt=0
while not isComplete(prime) and not flag:
    for i in range(P, N+1, P):
        if prime[i]!=0:
            prime[i]=0
            cnt+=1
            print(prime, cnt)  
            if cnt == K:
                flag=True
                result=i
                break
    P+=1
    
print(result)

