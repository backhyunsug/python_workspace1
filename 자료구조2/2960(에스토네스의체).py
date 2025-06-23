"""
2부터 N까지 모든 정수를 적는다.
아직 지우지 않은 수 중 가장 작은 수를 찾는다. 
이것을 P라고 하고, 이 수는 소수이다.
P를 지우고, 아직 지우지 않은 P의 배수를 크기 순서대로 지운다.
아직 모든 수를 지우지 않았다면, 다시 2번 단계로 간다.
"""

N = 30
K = 7
prime =[ x for x in range(0, N+1)]
prime[0]=0  #0번방은 안쓰겠다   # 
prime[1]=0
P=2 #첫번째 P값 

#P 이후에 다 0으로 채워진건지 확인 
def isComplete(prime, P):
    for i in range(P, len(prime)):
        if prime[i]!=0:
            return False #P이후에 하나라도 0이면 바로 리턴하기 
    return True #전체 다 끝나면 완료함 
 
#1 2 3 4 5 6 7 
#0 0   0   0

flag = False
result=0
cnt=0
#아직 모든 수를 지우지 않았다면,
#while 문안에서 for문이 작동중 for문안에서 종료조건을 알게 될경우 
#차라리 함수면 return 을 호출하면 return 함수종료문이기 때문에 
# for도 끝나고 while도 끝난다. 문제는 가끔 loop 안에 loop가 있는데 
# 내부 loop에서 break 를 호출하면 내부 loop만 종료한다. 
# 외부 loop는 의미가 없음  
while not isComplete(prime, P) and not flag:
    for i in range(P, N+1, P): #배수를 찾아서 0으로 초기화
        if prime[i]!=0: 
            prime[i]=0
            cnt+=1
            print(prime, cnt)  
            if cnt == K:
                flag=True #종료조건을 만족해서 내부 loop가 끝남 
                result=i
                break
    P+=1
    
print(result)

