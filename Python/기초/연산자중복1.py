#복소수 3+2i  2+3i  
class MyType:
    def __init__(self, x=0, y=0):
        self.x = x 
        self.y = y

    #연산자중복 - 이미 정해져 있음, 이름이 정해져있고
    #반환형태와 매개변수도 정해져 있다 
    # m3 = m1 + m2 
    # m3 = m1.__add__(m2)
    # result = m2 + m1  
    # m2.__add__(m1)  
    # self - 클래스 메서드들은 객체 자신에 대한 참조로 누구나 
    # self를 가져야 한다. other전달받은 매개변수 
    # 반환값이 객체여야 한다 
    def __add__(self, other):
        #print("********")
        result = MyType()
        result.x = self.x + other.x 
        result.y = self.y + other.y 
        return  result    

    #반환값 문자열이어야 한다 
    def __str__(self):
        return f"x:{self.x} y:{self.y}"
        #print(객체)
    #__sub__, __mul__ , __truediv__
    def __sub__(self, other):
        #print("********")
        return  MyType(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        #print("********")
        return  MyType(self.x * other.x, self.y * other.y)
         

m1 = MyType(4,5)
m2 = MyType(8,9)
#m3 = m1.append(m2)
m3 = m1.__add__(m2)
print(m3)
m3 = m1 + m2
print(m3)
print(m1.__sub__(m2).__str__())

print(m1 - m2)
print(m1 * m2)
#print(m1 / m2)

class MyInt(int):
    pass 

i = MyInt()

