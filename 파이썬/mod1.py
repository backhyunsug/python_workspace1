def add(x, y):
    return x+y
 
def sub(x, y):
    return x-y 

class Person:
    def __init__(self, name="", age=0):
        self.name = name
        self.age = age 

    def print(self):
        print(f"name={self.name} age ={self.age}")



#모듈은 내부에 내장변수로 __name__ 이라는 변수가 있다 
#이 변수는 이 파일이 직접 실행될때는 __main__으로 오고 
#import 되어서 실행될때는 파일명(mod1)으로 온다
#print("모듈명 ", __name__)

if __name__ == "__main__":
    print( add(3,4) )
    print( sub(3,4) )
    
    p1 = Person("조승연", 30)
    p1.print()

