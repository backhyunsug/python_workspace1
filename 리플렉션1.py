# 리플렉션 - 거울 
# 클래스를 만들면 , 언어 번역가들이 클래스 읽어서 정보를 해석해서 저장 
# 그 정보를 접근할 수 있게 해준다.
# 프레임워크 만들때   사용자가 클래스를 설계함 
# a = 클래스명() 

class Person:
    def __init__(self, name="", age=20):
        self.name= name  #두개의 필드가 있다 
        self.age = age 

    def greet(self):
        print(f"Hello {self.name}")

p = Person("Tom", 12)

#클래스내의 속성을 가져온다 
a = getattr(p, 'name') #특정 객체로부터 속성을 가져올 수 있다 
print(a)

a = getattr(p, 'age')
print( a)

print( dir(p)) #p 클래스 내부에 있는 구조 

#필터링을 해서 사용자가 만든거만 
fields = [x for x in dir(p) if not x.startswith("__")]
print(fields)

import inspect  #이 라이브러리가 각 요소가 함수인지 아닌지 확인가능 
for field in fields:
    print( getattr(p, field))

#특정 객체안에 있는 모든 메서드와 변수들을 다 가져온다 
print( inspect.getmembers(p) )
for item, value in inspect.getmembers(p):
    if inspect.ismethod(value) or inspect.isfunction(value):
        print("함수", item)
    else:
        print("변수", item) 
#[ 출력변수 for  출력변수 in iterabletype if 조건식]
# not( a or b) not a and not b   

#클래스로부터 변수이름만 추출하기 
var_fields = [ name for name, value in inspect.getmembers(p) 
    if  not( inspect.ismethod(value) or inspect.isfunction(value) )
     and not name.startswith("__") ]
print( var_fields )

#메서드 이름만 추출하기
fun_fields = [ name for name, value in inspect.getmembers(p) 
    if  ( inspect.ismethod(value) or inspect.isfunction(value) )
     and not name.startswith("__") ]
print( fun_fields )

a = getattr(p, var_fields[0])
print( a )

a = getattr(p, var_fields[1])
print( a )

#객체안의 변수의 값을 바꿀 수 도 있다 
setattr(p, 'name', '홍길동')
setattr(p, 'age', 43)

print (p.name, p.age)

def add(x, y): #함수도 주소임, 파이썬은 변수들한테 주소 저장가능 
    return x+y 
a = add 
print( a(5,6) )

method = getattr(p, "greet")
method() 

#함수의 매개변수 
params =  inspect.signature(add)
print(params)
print(params.parameters)


