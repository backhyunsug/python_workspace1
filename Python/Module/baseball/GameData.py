import random 

class Baseball:
    def __init__(self):
        self.computer = [-1, -1, -1, -1] #아무값도 없다 
        self.person = [-1, -1, -1, -1] #아무값도 없다
        self.count=0 #몇번했는지를 저장하기 위한 변수 
        self.personList=[]
    
    def init_computer(self):
        cnt=1 #랜덤값 3개를  추출해야 하는데 중복되면 안됨 
        while cnt<=3:
            v = random.randint(0,9)
            if v not in self.computer:#중복아닐때
                self.computer[cnt]=v 
                cnt +=1
          
    def init_person(self):
        s = input("숫자 3개(0~9사이의)를 입력하세요(예시 0 1 2)")
        numberList = s.strip().split(" ")
        self.person[1]=int( numberList[0])
        self.person[2]=int( numberList[1])
        self.person[3]=int( numberList[2])

    def getResult(self):
        #수트라이크, 볼, out개수 
        strike=0
        ball=0
        out=0

        for i in range(1, 4):         
            if self.person[i] in self.computer :
                if self.computer[i] == self.person[i]:
                    strike+=1
                else:
                    ball+=1  
            else:
                out+=1
        return strike, ball, out 

    def start(self):
        #3strike 이거나 5번의 기회를 다 사용했을경우에 종료한다 
        flag = False #아직 3strike가 아님을 나타내기위한 변수 
        self.init_computer()
        print(self.computer) 
        while  flag==False and self.count<=5:
            self.init_person()
            strike, ball, out = self.getResult()
            print(f"strike:{strike} bal:{ball} out:{out}")
            self.personList.append( 
                 {"person":[x for x in self.person], 
                "strike":strike, "ball":ball, "out":out})
            if strike ==3:
                flag=True 
            self.count+=1


if __name__ == "__main__":
    b = Baseball()
    # b.init_computer()
    # b.init_person()
    # print(b.computer)
    # print(b.person)
    # print(b.getResult())    
    b.start()

            



