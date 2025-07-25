#모듈 Weekpay.py 파일에 있는 클래스 Weekpay 를 가져오겠다.
from Weekpay import Weekpay #외부파일(모듈을) 이 파일로 불러오기 

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

    def search(self):
        name = input("찾을이름 : ")
        resultList = list(filter( lambda w : name in w.name, self.wList))
        if len(resultList) == 0:
            print("데이터가 없습니다")
            return 
        
        #resultList[0].output() #Weekpay의 output
        for w in resultList:
            w.output()

    def modify(self):
        name = input("찾을이름 : ")
        resultList = list(filter( lambda w : name in w.name, self.wList))
        if len(resultList) == 0:
            print("데이터가 없습니다")
            return 
        #enumerate함수는 인덱스랑, 데이터를 한꺼번에 반환한다 
        #resultList[0].output() #Weekpay의 output
        for i, w in enumerate(resultList):
            print(i, end ="\t")
            w.output()

        sel = int(input("수정할 대상을 입력하세요(숫자로)"))
        temp = resultList[sel]
        temp.name = input("이름 : ")
        temp.work_time = int(input("근무시간 ")) 
        temp.per_pay = int(input("시간당급여액 ")) 
        temp.process()

    def start(self):
        print("start")


if __name__ =="__main__":
    mgr = WeekPayManager()
    #mgr.output()
    #mgr.search()
    mgr.modify()  # . 도트연산자 - 소유권을 의미한다. 
    mgr.output()



