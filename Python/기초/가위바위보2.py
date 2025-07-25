"""
객체지향 - 클래스(사용자가 만드는 데이터 타입)
int, str, float -파이썬의 기본 타입들 하고 마찬가지 
기본 타입과 다른거는 
변수 = 클래스명()   클래스를 가지고 객체를 만든다. 객체는 heap공간에 
                  저장되고, 객체의 주소(참조)를 변수에 전달한다.
                  메모리가 부족하면 None전달  
객체 내부의 변수나 함수에 접근하려면 .(도트 연산자)를 이용해 접근한다 
생성자 __init__ 라는 이름을 갖고 기본적으로 파이썬에서는 개체의 자신의
주소를 전달하기 위해서 self라는 매개변수를 첫번째 매개변수로 가지고 다녀야 한다
변수명이 self일 필요는 없다. 그런데 남들도 self를 사용하니까 self를 사용하자

추상화 - 클래스 : 내부 구조를 몰라도 쓰는데 지장상황이 없다 
        추상화 <-> 구체화
        추상화 될수록 사용자나 편하다 반대로 
        추상화를 할수록 만드는 사람은 없다. 
        누군가 한사람을 죽도록 고생시켜서 모두가 행복해진다.
        클래스 설계 패턴 - 32가지(디자인패턴) 
은닉화 - 다른언어에서는 은닉화를 많이 지원한다. 
        접근권한을 만들어서 특정변수나 함수에 외부로 부터 접근불가상황이 기본
        외부에서 접근하게 별도로 권한을 줘야 한다. 
        Private, public  - 파이썬은 기본이 public 만약에 
        일부 변수가 외부에 노출되기 싫어 __(언더바 2개)를 변수나 함수앞에
        붙이면 priavate로 안다.
        WeekPay - 함수하나도 안만들고 변수만 있어도 된다.
        WeekPayManager - 이 클래스를 통해서 접근, 여기서만 
           wList 파이썬에서는 최소화
상속성 - X
다형성 - __init__ 매개변수 기본값 
"""            
#가위바위보 게임한판 - computer, person, 승부여부 
import random 

class GameData:
    #변수선언을 생성자에서 하자 
    # : 이유, 그래야 객체가 생성될때마다 새로운 메모리를 만들어준다
    def __init__(self):
        self.computer = 0
        self.person= 0
        self.winner=0
    
    def gameStart(self):
        self.computer = random.randint(1,3)
        self.person = int(input("1.가위 2.바위 3.보"))
        self.winner = self.isWinner() 

    def isWinner(self):
        s = self 
        if s.computer == s.person:
            return 3 # 무승부 
        #명령어가 한문장 이상일때 연결하는 문자 \ 양쪽에 공백 필요함 
        if (s.computer==1 and s.person==3) or  \
           (s.computer==2 and s.person==1) or  \
           (s.computer==3 and s.person==2) : 
            return 1 #컴퓨터승
        
        return 2 #사람승

    def printLog(self):
        print(f"컴퓨터:{self.computer} 사람:{self.person} 승부:{self.winner}")


class Game:
    titles1=["", "가위", "바위", "보"]
    titles2 =["", "컴퓨터승", "사람승", "무승부"]

    def __init__(self):
        self.gameList = []  #game 정보를 저장한다 
    
    def printLog(self, g):
        print(f"컴퓨터: {self.titles1[g.computer]}", end="\t")
        print(f"사람: {self.titles1[g.person]}",   end="\t")
        print(f"승부: {self.titles2[g.winner]}" )
        

    def start(self):
        while True:
            g = GameData()
            g.gameStart()
            #g.printLog()
            self.printLog(g) 
            self.gameList.append( g )

            again=input("1.계속 0.종료? ")
            if again != "1":
                return 

    def printResult(self):
        print(f"{len(self.gameList)}번 수행함")
        for g in self.gameList:
            self.printLog(g)

    def mainStart(self):
        self.start()
        self.printResult()

if __name__ == "__main__":
    # g = GameData() #객체 생성
    # g.gameStart()  #g -> self로 전달된다다
    # g.printLog() 
    game = Game()
    game.mainStart()



