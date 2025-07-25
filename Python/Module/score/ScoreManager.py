from ScoreData import ScoreData
import pickle 

#ScoreData.py 파일에서 ScoreData 클래스를 가져와라 

class ScoreManager:
    def __init__(self): #생성자-파이썬에서는 변수도 만들고 
                        #첫 시작시 준비작업 
        self.scoreList = [
            # ScoreData(),
            # ScoreData("조승연", 90, 80, 90),
            # ScoreData("안세영", 80, 80, 70),
            # ScoreData("김연경", 90, 90, 90),
            # ScoreData("김연아", 100, 80, 100)
        ]
    
    def printAll(self):
        for s in self.scoreList:
            s.print()

    def menuDisplay(self):
        print("--------")
        print("  메뉴   ")
        print("--------")
        print("1. 추가  ")
        print("2. 출력  ")
        print("3. 검색  ") #이름
        print("4. 수정  ") #이름
        print("5. 삭제  ") #이름
        print("6. 정렬  ") #총점 내림차순으로
        print("7. 저장  ")
        print("8. 불러오기") 
        print("0. 종료  ")
        print("--------")
    
    def save(self):
        with open("score.bin", "wb") as f:
            pickle.dump(self.scoreList, f)

    def load(self):
        with open("score.bin", "rb") as f:
            self.scoreList = pickle.load(f)
        self.printAll()


    def append(self):
        sc =  ScoreData() #객체 생성 
        sc.name = input("이름 : ")
        sc.kor = int(input("국어 : "))
        sc.eng = int(input("영어 : "))
        sc.mat = int(input("수학 : "))
        sc.process()
        self.scoreList.append(sc)

    def search(self):
        name = input("이름 : ")
        #filter는 두번째 매개변수로 전달된 list를 받아서 
        #for문 돌려서 첫번째 매개변수로 전달된 함수를 호출 
        #람다 : 매개변수하나(scoreList에 저장된 객체 하나 )
        #      반환은 True 또는 False 
        # 매개변수 ScoreData객체 
        #전체 실행하는게 아니라 실행준비상태임 for를 사용하거나 
        #list로 둘러 쌓으면 list생성자가 호출되면서 filter가 모든 작업을
        #완료한다. 
        resultList = list(filter(lambda item: name in item.name, 
                                self.scoreList))
        #데이터가 없을 경우에 처리 len(resultList) 데이터 개수 반환
        if len(resultList)==0:
           print("찾으시는 데이터가 없습니다.")
           return #else 사용하지 말고 함수종료 

        #enumerate함수가 list 를 전달하면 index와 객체 tuple을 반환
        for i, s in enumerate(resultList):
            print(f"[{i}] ", end=" ")
            s.print()

    def  modify(self):
        name = input("수정할 이름 : ")
        resultList = list(filter(lambda item: name in item.name, 
                                self.scoreList))
        #데이터가 없을 경우에 처리 len(resultList) 데이터 개수 반환
        if len(resultList)==0:
           print("찾으시는 데이터가 없습니다.")
           return #else 사용하지 말고 함수종료 

        #enumerate함수가 list 를 전달하면 index와 객체 tuple을 반환
        for i, s in enumerate(resultList):
            print(f"[{i}] ", end=" ")
            s.print() 
        sel = int(input("수정할 대상은(번호) "))
        #수정할 대상의 참조를 가져온다 
        item=self.resultList[sel]
        item.name = input("이름 : ")
        item.kor = int(input("국어 : "))
        item.eng = int(input("영어 : "))
        item.mat = int(input("수학 : "))
        item.process() #다시 계산하기 
    
    def  delete(self):
        name = input("삭제할 이름 : ")
        resultList = list(filter(lambda item: name in item.name, 
                                self.scoreList))
        #데이터가 없을 경우에 처리 len(resultList) 데이터 개수 반환
        if len(resultList)==0:
           print("찾으시는 데이터가 없습니다.")
           return #else 사용하지 말고 함수종료 

        #enumerate함수가 list 를 전달하면 index와 객체 tuple을 반환
        for i, s in enumerate(resultList):
            print(f"[{i}] ", end=" ")
            s.print() 
        sel = int(input("삭제대상은? " ))
        #remove 객체 참조를 직접 부여한다. 그 객체를 찾아서 삭제 
        self.scoreList.remove(resultList[sel]) #-실제 대상삭제
        #del resultList[sel] #삭제 -- 명단삭제
    
    def sort(self):
        #원본 냅두고 정렬한 결과만 출력하기
        #key 에 전달해야할 람다는 
        #매개변수하나 반환값 정렬을 할수있는 데이터타입 
        # > < 연산자가 가능한다 
        #s1 = ScoreData()
        #s2 = ScoreData()
        #s1 > s2 - 
        #파이썬이 제공하는 기본타입들 int, float, str, ...

        resultList = sorted(self.scoreList, key=lambda item : item.total 
               , reverse=True)
        for i in resultList:
            i.print() 

    def start(self):
        #함수주소를 배열에 저장하고 호출함 ,
        funcList =[None, self.append, self.printAll, 
                   self.search, self.modify, 
                   self.delete, self.sort, 
                   self.save,   self.load]    
        while True:
            self.menuDisplay()
            choice = int(input("선택 : "))
            if choice>0 and choice<len(funcList):
                funcList[choice]()
            elif choice==0:
                return 
            else:
                print("잘못된 메뉴입니다.")


if __name__ =="__main__":
    sm = ScoreManager()
    sm.start()








