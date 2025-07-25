#https://school.programmers.co.kr/learn/courses/30/lessons/120837
def solution(hp):
    answer = 0
    ant=[5,3,1]
    i=0
    while i<3:
        mok = hp//ant[i] 
        answer+=mok 
        hp = hp%ant[i]
        i+=1 
        
    answer += hp 
    return answer

print( solution(999) )