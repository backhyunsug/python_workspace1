def isEven(n):
    if n%2==0:
        return True 
    return False 

def toUpper(s):
    # A - 65     a - 97 둘사이에 32만큼   소문자를 대문자로 -32  대문자를 소문자로 +32 
    temp=""
    for c in s:
        if ord(c)>=ord('a') and ord(c)<=ord('z'): #소문자일때 
            c = chr(ord(c)-32)  # 'a' - 97 -32 = 65 => 문자로 바꾼다 
        temp = temp + c
    return temp   