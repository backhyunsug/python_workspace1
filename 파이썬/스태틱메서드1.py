#사칙연산을 하는 클래스를 만들어 보자 
#공통위 데이터 공간을 두고 
class Calculator:
    def __init__(self, x=0, y=0):
        self.x = x #데이터 
        self.y = y
    def add(self): #함수
        return self.x+self.y 
    def sub(self):
        return self.x-self.y 

c1 = Calculator(4, 5)
print(c1.add())
print(c1.sub())

class Calculator2:
    def add(self, x, y):
        return x+y 
    
    def sub(self, x, y):
        return x-y 
    
c1 = Calculator2()
print( c1.add(4,5))
print( c1.sub(4,5))
#staicmethod는 객체와 아무 관계없다
#self도 매개변수로 갖지 못한다 
#사용목적이 객체 안만들고 특정메서드를 사용하고 싶다
class Calculator3:#자바-어노테이션  
    @staticmethod #@staicmethod-데코레이터 
    def add(x, y):
        return x+y 

    @staticmethod
    def sub(x, y):
        return x+y 
    
    #cls 를 사용하던 말던 매개변수로 전달은 반드시 필요하다
    @classmethod
    def mul(cls, x, y):
        return x*y 
    
print( Calculator3.add( 4, 5) )
print( Calculator3.sub( 4, 5) )
print( Calculator3.mul( 4, 5) )

#Math가 수학함수들 갖고 있음, cosin, sin, round .....
#웹개발을할때  게시글 : <h1>Hi Hello</h1>
# 디비에 접근해야하는 코드를 각각의 클래스가 소유할 경우 문제점 
# 1. 디비 아이피 바뀌었을때 , 아이디 패스워드 바뀌면 
#    모든 클래스를 다 바꾸는 문제가 발생 , 보안도 걸린다.
#    패스워드가 드러나게하면안된다. 
# staticmethod나 classmethod로 구성된 클래스를 만들어서 
# 사용하는것이 바람직 하다. 
# staticmethod단점이 클래스 변수를 못건드린다. 
# classmethod를 사용한다 
# staticmethod는 함수들간에 기능적 유기성은 있지만 데이터가 
# 필요없을때 유용하다 
# classmethod는 매개변수 cls 를 갖고 다닌다. 
# class변수에 접근가능하다. 



