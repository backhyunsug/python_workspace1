#컴프리핸션 
a = [1,2,3,4,5,6,7,8,9,10]
b = a 
a[2] = -3 

print( a )          #1.둘이 같다  2.둘이 다르다 
print( b )

#스택 하고 힙을 시스템이 프로세스한테 할당한다. 
#스택에는 변수 자신이 저장된다
#힙에는 데이터가 저장된다. 저장된 주소를 변수한테 전달한다 
# a(100번지) ========> (1,2,3,4,5,6,7,8,9,10)
# b = a  : 동일한 데이터 공간을 공유한다 
# b(100번지지)
# b라는 메모리 공간을 스택에 만들고 a의 값(데이터가있는곳의주소)
#를 복사한다. 소프트카피, 얖은카피라고 부른다
#소프트카피의 목적은 메모리 절약,쓸데없는 복사과정(overhead) 이 필요없다.
#하드카피, 깊은카피의 경우는 직접  구현하거나 deepcopy 모듈을 사용하거나 컴프리헨션을 사용한다

#사용자가 구현한 하드카피 상황
b = [] #새로 기억공간을 만든다 
for item in a: #a 로 부터 데이터를 하나씩 가져와서 item이라는 변수에 저장한다 
    b.append(item) 
b[3]=99
print(a)
print(b)

#컴프리핸션    : 리스트 복사   [ 변수명 for 변수명 in 리스트형변수]

c = [item for item in a] #하드카피 
c[5] = 55
print("a= ", a)
print("c= ", c)

d = [ item*2 for item in a] 
print("d= ", d)

#조건을 부여할 수도 있다. [변수명 for 변수명 in 리스트형변수   if 변수명> 0 ]
#짝수와 홀수를 나눠보자 a의 숫자를 
oddList = [ x for x in a if x%2==1 ]
print( oddList )

wordList = ["rain", "desk", "hospital", "building", "java", "python", 
            "cloud", "rainbow", "assembly", "javascript", "html", "css"]
#1.복사 - 하드카피 
wordList2  = [w for w in wordList]
print( wordList2 )

#2.복사하면서 대문자로 바꾸고 싶다
wordList3 = [w.upper() for w in wordList]
print( wordList3 )

#단어와 단어길이
wordList3 = [ (w, len(w), w.upper()) for w in wordList]
print( wordList3 )

#단어길이가 5글자 이상인 것만 
wordList3 = [w for w in wordList if len(w)>=5]
print( wordList3 )

#문제1. 단어중에 java 라는 단어가 있는것만 추출하기 
wordList3 = [w for w in wordList if 'java' in w]
print( wordList3 )

#문제2. 단어중에 길이가 5개보다 짧은단어만 추출하기 
wordList3 = [w for w in wordList if len(w)<5]
print( wordList3 )
