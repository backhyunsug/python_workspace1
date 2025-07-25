def getFactor(n):
    count = 0
    for i in range(1,int(n**0.5)+1):
        if n % i == 0:
            if i == n // i:
                count += 1
            else:
                count += 2    
    return count

def solution(number, limit, power):
    answer = 0
    nList = []
    for i in range(1,number+1):
        nList.append(getFactor(i))
        
    for i in range(0,len(nList)):
        if nList[i] > limit :
            nList[i] = power
            answer+= nList[i]
        else:
            answer+= nList[i]

    return answer
    