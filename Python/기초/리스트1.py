#성적처리 3명이면 
#name1, name2, name3
#list타입 => 배열 
words = ["red", "green", "blue"]  #list타입, 인덱싱과 슬라이싱을 지원한다 
print( words[0] ) #indexing 
print( words[1] )
print( words[2] )
print( words ) #한번에 출력가능 
#새로운 단어 추가하기 
words.append("black")
words.append("cyan")
print( words )
print("단어개수", len(words))
print("red개수", words.count("red"))
print("red위치", words.index("red"))
if words.count("yellow") :
    print("yellow 위치", words.index("yellow"))
else:
    print("yello는 없다")

#in연산자   "내용" in list타입  있으면 True 없으면  False 를 반환한다 
if  "yellow" in words :
    print("yellow 위치", words.index("yellow"))
else:
    print("yello는 없다")

print( words[::-1]) 

#인덱싱- list타입의 경우에는 인덱싱을 통해 값 변경가능 , 문자열은 indexing을 통한 값 변경불가  
words[0]="white"
print( words )

s = "white"
#s[0] ='W' 문자열의 경우는 인덱싱을 통한 값 변경은 불가능하다. 
s = s.replace("w", "W")
print( s )
s = "white"
s2 = "W"+s[1:]
print( s2)

#extend  함수 - 리스트와 리스트를 합친다. 
words.extend( ["brown", "violet", "puple", "magenta"])
print( words)

#list => str , join을 사용한다 
s = ", ".join(words)
print(s)

#str => list 
words2 = s.split(", ")
print( words2 ) 

numbers = [1,2,3,4,5,6,7,8,9,10]
print(numbers[0])
print(numbers[0::2])
print(numbers[::-1])
print(numbers[1::2])

#리스트만들기 
names = [] # names =list()  동일한 문법이다
names.append("홍길동")
names.append("임꺽정")
names.append("장길산")
names.append("홍경래")
print(names) 

names =list()
names.append("모란")
names.append("작약")
names.append("불두화")
names.append("목련")

print(names)


