class MyType:
    def __init__(self, x=0, y=0):
        self.x = x 
        self.y = y 
    def add(self):
        return self.x+self.y 
    
    def sub(self):
        return self.x-self.y 
    
    def mul(self):
        return self.x*self.y 

m1 = MyType() 
import inspect 
var_fields = [name for name, value in inspect.getmembers(m1) 
              if not (inspect.ismethod(value) 
                      or inspect.isfunction(value))
                and not name.startswith("__")]
print(var_fields)
fun_fields = [name for name, value in inspect.getmembers(m1) 
              if  (inspect.ismethod(value) 
                      or inspect.isfunction(value))
                and not name.startswith("__")]
print(fun_fields)
setattr(m1, var_fields[0], 10)
setattr(m1, var_fields[1], 5)
print(m1.x, m1.y)
print( getattr(m1, fun_fields[0])())
print( getattr(m1, fun_fields[1])())
print( getattr(m1, fun_fields[2])())

#문제 1. inspect 써서 변수 리스트 
#문제 2. inspect 써서 함수 리스트 
#문제 3. setattr 써서 x에는 10 y에는 5 
#문제 4. getattr로 함수 주소 갖고 와서 호출하기 