#가위바위보 게임 :통계 
import random #import 외부 파일(모듈)을 이 파일로 갖고 들어와라 
#거듭해서 import를 하더라도 한번만 들어온다. 
#게임게시 : gameStart() 

# 출력값  : 컴퓨터이김 1, 사람이김 2 : 무승부 3 판단하는 함수 
# 입력값  : 매개변수는 컴퓨터의 생각, 사람의입력값 

def isWinner(computer, person):
    if computer == person: #사람하고 컴퓨터가 같다 
        return 3  #무승부임 
    
    #컴퓨터가 이기는 경우의 수 or
    # 가위-1 바위-2 보-3 
    # \ 기호는 여러줄에 걸쳐서 문장을 작성할때 같은줄이다라는 의미 \ 앞뒤로 
    # 공백이 필요하다  
    if (computer==1 and person==3) or \
       (computer==2 and person==1) or \
       (computer==3 and person==2):
        return 1 #컴퓨터가 이김 
    
    #사람이 이기는 경우의 수 or .....
    return 2 

titles = ["", "가위", "바위", "보"]
titles2= ["", "컴퓨터승", "사람승", "무승부"]
gameList=[] #{"computer":"값", "person":"값", winner:"값"}

def test():
    for i in range(0, 10):
        computer = random.randint(1,3)
        person = random.randint(1,3)
        winner = isWinner(computer, person)
        print("컴퓨터:", titles[computer], "사람:",titles[person], titles2[winner])

def gameStart():
    gameList.clear() #데이터만 삭제 
    #계속 반복 
    while True:
        computer = random.randint(1,3)
        person = int(input("1.가위 2.바위 3.보 "))
        winner = isWinner(computer, person)
        print("컴퓨터:", titles[computer], "사람:",titles[person], titles2[winner])
        gameList.append({"computer":computer, "person":person, 
                         "winner":winner})
        again = input("게임을 계속 하시겠습니까?(y/n) ")
        if again !="Y" and again!="y":
            return  

def gameStatistic():
    computer_win=0
    person_win=0
    equal_win=0
    for game in gameList:
        if game["winner"]=="1":
            computer_win+=1 
        elif game["winner"]=="2":
            person_win+=1 
        else:
            equal_win+=1 

    for game in gameList:
        print(f"컴퓨터:{game["computer"]}", end="\t")
        print(f"사람:{game["person"]}", end="\t")
        print(f"승패:{game["winner"]}")
    
    print("컴퓨터 승 ", computer_win)
    print("사람 승 ", person_win)
    print("무승부 ", equal_win)
    
        
#gameStart() #함수호출을 하자 
def gameMain():
    
    while True:
        print("1.게임시작")
        print("2.게임통계")
        print("3.게임종료")
        sel = input("선택 : ")
        if sel=="1": 
            gameStart() 
        elif sel=="2":
            gameStatistic()
        elif sel=="3":
            print("게임을 종료합니다")
            return 
        
gameMain()
