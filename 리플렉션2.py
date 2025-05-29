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
    
#문제 1. inspect 써서 변수 리스트 
#문제 2. inspect 써서 함수 리스트 
#문제 3. setattr 써서 x에는 10 y에는 5 
#문제 4. getattr로 함수 주소 갖고 와서 호출하기 