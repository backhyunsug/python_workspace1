#https://school.programmers.co.kr/learn/courses/30/lessons/12951?language=python3

def solution(s):
    s = s.lower() #전체를 소문자로 바꾸고 
    answer = ''

    for i in range(0, len(s)):
        if s[i]>="0" and s[i]<="9": #숫자일 경우 그냥 붙인다 
            answer += s[i]
        elif s[i]==" ":
            answer += s[i]            
        else:
            #문장 첫번째 글자였거나   자기 앞에 글자가 공백이면 대문자로 바꿔서 
            if i==0 or i>0 and s[i-1]==" ":
                answer += s[i].upper()
            else:
                answer += s[i]    
        
    return answer

print ( solution("for the last week") )
# 3people unFollowed me"	"3people Unfollowed Me"
# "for the last week"	"For The Last Week"
