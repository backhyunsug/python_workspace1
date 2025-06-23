class MyClass:
    #클래스변수의 영역이다. 
    #객체를 만들던 말던 한번만 만든다.
    #생성자에서 이 부분을 건드리면 안된다.
    count=0  #객체가 만들어질때마다 몇개 만들어졌는지 확인을 
             #하고 싶다  
    @staticmethod
    def addCount():#static메서드 입장에서는 매개변수
        count+=1  #인 self를 사용못하니까 count변수 접근불가
    
    @classmethod 
    def increase(cls): #cls는 클래스다 
        cls.count += 1 
MyClass.increase()
MyClass.increase()
print(MyClass.count )

#print( MyClass.addCount() ) --error

#객체를 만들때 객체의 개수를 카운트하거나 제한하는 클래스를 
#만들고자 할때 classmethod를 사용한다. 
 
class SelfCount:
    #변수에 __를 붙이면 외부에서 접근이 불가능하다
    __count = 0  #클래스변수를 선언한다 
    #생성자에서 값 증가하기 
    def __init__(self):
        SelfCount.__count+=1 
        print(SelfCount.__count)

    @classmethod
    def count_output(cls):
        print(cls.__count)

s1 = SelfCount()
SelfCount.count_output()
s1 = SelfCount()
SelfCount.count_output()
s1 = SelfCount()
SelfCount.count_output()
s1 = SelfCount()
SelfCount.count_output()


#print( SelfCount.__count ) #이속성을 볼수없다라는 에러메시지 
#


       