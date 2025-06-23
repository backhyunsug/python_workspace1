#https://school.programmers.co.kr/learn/courses/30/lessons/12977
#
def isPrime(num): #소수인지 물어봄    1 과 자기자신으로만 나눠짐 
    i=2
    while i<=num/2:  # 1 2 3 4 5 6 7 8 9 .. 절반까지만  
        if num%i==0:  
            return False  #소수가 아니라는걸 판단한 순간 함수가 종료  
        i+=1
    return True #마지막까지 남으면 소수 

#1,2,7,6,4    1  2 7 6 4     1 2 4 6 7 

def solution(nums):
    answer = 0

    for i in range(0, len(nums)-2):
        for j in range(i+1, len(nums)-1): #중복 허용 안함 
            for k in range(j+1, len(nums)):
                if isPrime(nums[i]+nums[j]+nums[k]):
                   #print(nums[i],nums[j],nums[k])
                   answer+=1     
    return answer 

#print( isPrime(12))
print( solution([1,2,3,4]) )
