def solution(s):
    s = s.lower()
    answer = ''

    for i in range(0, len(s)):
        if ord(s[i])>=48 and ord(s[i])<=57: #숫자일 경우 그냥 붙인다 
            answer += s[i]
            i+=1 
        elif s[i]==" ":
            answer += s[i] 
            i+=1 
        else:
            if i==0 or i>0 and s[i-1]==" ":
                answer += s[i].upper()
            else:
                answer += s[i]    
        
    return answer

print ( solution("for the last week") )
# 3people unFollowed me"	"3people Unfollowed Me"
# "for the last week"	"For The Last Week"
