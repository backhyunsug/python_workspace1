for i in [1,2,3,4,5,6,7,8,9,10]:
    print(i)

#range(시작,종료,증감치)  시작값부터 시작해서 종료값-1까지 증감치를 가지고 생성해낸다 
print( range(1,11) )
for i in range(1,11):
    print(i) 

a = list(range(1,11))
print(a)

for kk in range(2, 100,2):
    print(kk, end=' ')
print()

for kk in range(100, 1, -2):
    print(kk, end=' ')
print()

#문제1. 1 2 3 4 5 6 7 8 9 10
#      11 12 13 14 15 ......
# 
#   
for i in range(1, 101):
    print(i, end=' ')
    if i%10==0:
        print() 
#문자의 unicode -> ord 
print( ord('A'))  #65
print( ord('B'))  #66
print( ord('a'))  #97
print( ord('0'))  #48
print( ord('1'))
#코드값 => 문자로 chr(코드값)
print( chr(49))
print( chr(65))

#for문을 써서 알파벳 A~Z까지 출력하기  

for i in range(ord('A'), ord('Z')+1):
    print( chr(i), end='')
print() 


#키보드로부터 문장을 입력받아서  각 문자의 개수 
#  A  ===> 5       대소문자 구분하지 않고 
#  B  ===> 0   조건식 and 조건식   조건식 or 조건식 

count =[0,0,0,0,0, 0,0,0,0,0,
        0,0,0,0,0, 0,0,0,0,0,
        0,0,0,0,0,0]

sentence = input("문장 : ")
for i in sentence: #한글자씩 가져온다
    if ord(i)>=ord('A') and ord(i)<=ord('Z'):
        count[ord(i)-ord('A')]+=1 
    elif ord(i)>=ord('a') and ord(i)<=ord('z'):
        count[ord(i)-ord('a')]+=1
    
for i in range(0,len(count)):
    print( chr(i+ord('A')), "===>" , count[i])
print() 
#dict 타입   'A' 존재안하면 하나 만들고 1 존재하면 + 1
result={}  #result=dict() 
for i in sentence.lower():
    if i in result.keys():
        result[i]+=1
    else:
        result[i]=1

for item in result.items():
    print(item)
