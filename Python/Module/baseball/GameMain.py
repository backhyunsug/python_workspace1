from GameData import Baseball

class GameMain:
    def __init__(self):
        self.gameList = []
    
    def start(self):
        while True:
            print("1.게임시작")
            print("2.통계")
            print("0.종료")
            sel = input("선택 : ")
            if sel =="1":
                self.gamestart() 
            elif sel=="2":
                self.showStatistics()
            else:
                return 
            
    def gamestart(self):
        b = Baseball()
        b.start() 
        self.gameList.append(b)
    
    def showStatistics(self):
        for b in self.gameList:
            print(b.computer)
            for item in b.personList:
                print(item["person"], item["strike"], 
                       item["ball"], item["out"], b.count)


if __name__ == "__main__":
    g = GameMain()
    g.start()