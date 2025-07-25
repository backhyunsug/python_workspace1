class Weekpay:
    def __init__(self, name="", work_time=20, per_pay=10000):
        self.name = name 
        self.work_time=work_time
        self.per_pay=per_pay
        self.process() #클래스 내부 함수 호출시 self로 해야 한다 

    def process(self):
        self.pay = self.work_time * self.per_pay

    def output(self):
        print(f"{self.name} {self.work_time} {self.per_pay} {self.pay}")

"""
w1 = Weekpay("홍길동", 20, 20000)
w1.output()

wList = [
    Weekpay("홍길동", 20, 20000),
    Weekpay("고길동", 10, 50000),
    Weekpay("김길동", 30, 40000),
    Weekpay("이길동", 40, 20000),
    Weekpay("장길동", 20, 20000)
]

for w in wList:
    w.output()
"""

class WeekPayManager:
    
    def __init__(self):
        self.wList = [
            Weekpay("홍길동", 20, 20000),
            Weekpay("고길동", 10, 50000),
            Weekpay("김길동", 30, 40000),
            Weekpay("이길동", 40, 20000),
            Weekpay("장길동", 20, 20000)
        ]

    def output(self):
        for w in self.wList:
            w.output()


mgr = WeekPayManager()
mgr.output()





