word = "racecar"
if word == word[::-1]:
    print(True)
else:
    print(False)

#절반을 쪼개서 - 길이 
#길이 ln len(문자열길이)-1 
#0        ln-0          arr[0] <-> arr[ln-0] 
#1        ln-1          arr[1] <-> arr[ln-1]
#2        ln-2          arr[2] <-> arr[ln-2]
#3        ln-3          arr[3] <-> arr[ln-3]

import math
def palindrom(s):
    ln = len(s)-1 
    #6   6/2 - 3 
    for i in range(0, math.ceil(ln/2)):
        if s[i] != s[ln-i]:
            return False 
    
    #마지막까지 남았다는 말은 회문이 성립한다
    return True 

print( palindrom("madam") )
print( palindrom("tomato") )
print( palindrom("abba") )