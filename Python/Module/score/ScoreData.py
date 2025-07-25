
#한사람정보 - 데이터베이스 레코드 하나 
#파이썬의 경우는 파일명과 클래스명은 아무 관계없다. 
class ScoreData:
    def __init__(self, name="홍길동", kor=100, eng=100, mat=100):
        self.name = name 
        self.kor = kor 
        self.eng = eng 
        self.mat = mat 
        self.process() 
    
    def process(self):
        self.total = self.kor + self.eng + self.mat 
        self.avg = self.total/3 
        if self.avg>=90:
            self.grade="수"
        elif self.avg>=80:
            self.grade="우"
        elif self.avg>=70:
            self.grade="미"
        elif self.avg>=60:
            self.grade="양"
        else:
            self.grade="가"

    def print(self):
        print(f"{self.name}", end="\t")
        print(f"{self.kor}", end="\t")
        print(f"{self.eng}", end="\t")
        print(f"{self.mat}", end="\t")
        print(f"{self.total}", end="\t")
        print(f"{self.avg:.2f}", end="\t")
        print(f"{self.grade}", end="\n")
        
if __name__ == "__main__":
    s = ScoreData()
    s.print()        
        