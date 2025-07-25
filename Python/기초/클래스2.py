class Person:
    #이 공간은 클래스 공간이다. 클래스 정의할때 딱 한번 실행된다. 
    #객체 만들때마다 실행되지 않는다. 그래서 list타입이나 dict타입등을 함부로 여기에 
    #선언하면 안된다. 
    name="홍길동"
    age=12
    phone=[] #공통공간 

    #생성자에서 변수를 만들자. 
    def __init__(self):
        self.name = ""
        self.age=0
        self.phone=[]

    def append(self, name="임꺽정", age=13, phone="010-0000-0001"):
        self.name = name 
        self.age = age
        self.phone.append( phone ) 

    def output(self):
        print(self.name, self.age, self.phone)

p1 = Person()
p1.append("장길산", 11, "010-0000-0003")

p2 = Person()
p2.append("김종서", 13, "010-0000-0004")

p1.output()
p2.output()

"""
주급 : 이름, 시간당급여액, 근무시간  => 객체지향으로 한사람분만 
"""