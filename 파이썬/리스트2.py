a = [1,2,3,4]
b = [5,6,7,8]

#리스트를 결합하는 방법, 원본을 안바꾸고 더해진 새로운 list를 반환 
c = a + b #연산자 중복기능 => 우리가 연산자를 새로 만들 수 있다. 
print(c)

a.extend(b) #원본에 추가가
print(a)

s = "hello"
if s == "hello": # if s.equals("hello") -- java
    print("같다")
else:
    print("다르다")

#오소삭제   del 삭제할요소 , del 객체
del c[0]  
print(c)

del c[4:] #4번째 요소 이후로 모두 삭제를 하라 
print(c)

a = [4,3,5,7,9,1,11,17, 12,15,8]
a.sort() #정렬 : 순서대로 데이터를 늘어놓는것을 말한다. 
#오름차순 정렬 : 갈수록 커지는 것  
#내림차순 정렬 : 갈수록 작아지는 것 
print(a)
a.reverse() #순서뒤집기
print(a)

#insert - 특정위치에 데이터 끼워넣기 
a.insert(0, 100) #0번째 위치에 값 100 넣어보기 
print(a)
a.insert(5, 77) #5번째 위치에 값 77을 넣어보기 
print(a)
a.insert(len(a), 88) #append 함수와 동일한 역할을 한다. 
print(a)

#a.remove(값)   - 값을 찾아서 (여러개일때는)첫번째로 나오는 값을 찾아서 삭제
a.remove(77)
print(a)

"""
pop함수가 필요했는지

컴퓨터안에 데이터를 저장하는 구조가 많다 
배열구조
링크드리스트 구조 

스택구조 - 후입선출(Last In First Out) - 나중에 들어간것이 먼저 나오는 구조이다
큐구조 - 선입선출(First In First Out)- 먼저 들어간것이 먼저 나오는 구조이다. 

"""
a=[]
a.append("A")  #스택구조의 push동작 데이터가 거꾸로 들어감 
a.append("B")
a.append("C")
a.append("D")
a.append("E")

print(a)
print(a.pop())
print(a)
print(a.pop())
print(a)
print(a.pop())
print(a)
print(a.pop())
print(a)
print(a.pop())
print(a)