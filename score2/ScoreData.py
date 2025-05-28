from sqlalchemy import text 
from DBEngine import theEngine 

#orm을 사용할때는 클래스를 만들어놓고 쓰는것이 맞음 
#
class ScoreData:
    #db에서 레코드셋 가져왔을때, ScoreData객체로 만들어서 
    #따로 관리할 수도 있다. 
    # select ... (kor+eng+mat) total  
    #s = ScoreData() 
    #s = ScoreData("홍길동")  .........
    def __init__(self, sname="", kor=0, eng=0, mat=0):
        self.sname = sname
        self.kor   = kor
        self.eng   = eng 
        self.mat   = mat
        self.process()  

    def output(self):
        print(f"{self.sname}", end="\t")
        print(f"{self.kor}", end="\t")
        print(f"{self.eng}", end="\t")
        print(f"{self.mat}", end="\t")
        print(f"{self.total}", end="\t")
        print(f"{self.average}", end="\t")
        print(f"{self.grade}")

    def process(self):
        self.total = self.kor+self.eng+self.mat 
        self.average = self.total/3 
        if self.average >= 90:
            self.grade = "수"
        elif self.average >= 80:
            self.grade = "우"
        elif self.average >= 70:
            self.grade = "미"
        elif self.average >= 60:
            self.grade = "양"
        else:
            self.grade = "가"        
        
if __name__=="__main__":
    with theEngine.begin() as conn:
        sql ="select * from tb_score"
        result = conn.execute(text(sql))
        for r in result.mappings().all(): #tuple로 가져온다 
            s = ScoreData(r["sname"], r["kor"], r["eng"], r["mat"])
            s.output()
            
            
