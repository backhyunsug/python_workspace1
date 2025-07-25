class Book:
    title="채식주의자"
    def __init__(self, title="쌍갑포차", paramprice=10000):
        print("Self", self) #객체주소 
        self.title = title        #self를 이용해서  
        self.price = paramprice
        self.count=10
        self.process() #함수도 매개변수로  self를 받아가야 한다

    def process(self):
        self.total_price = self.price * self.count 

    def output(self):
        print(self.title, self.price, self.count, self.total_price)
    pass 

b = Book() #객체가 만들어진다. 
#title이라는 클래스 내부 변수에 접근하려면 접근연산자로 .(도트)
print(b)
print(b.title)
print(b.price)  #b -> self로 전달 

b2 = Book("아 지갑놓고 나왔다")
print(b2)
print(b2.title)

b3 = Book("뽀짜툰", 30000)
print(b3)
print(b3.title)

