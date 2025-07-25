from DBEngine import theEngine
from sqlalchemy import text 
from ScoreData import ScoreData 

class ScoreManager:
    def __init__(self):
        pass

    def output(self):
        sql = "select * from tb_score"
        self.getList(sql)
        for s in self.scoreList:
            s.output() 


    def getList(self, sql):
        self.scoreList =[]
        with theEngine.begin() as conn:
            result = conn.execute(text(sql))
            for r in result.mappings().all():
                s = ScoreData(r["sname"], r["kor"], r["eng"], r["mat"])
                self.scoreList.append( s )

    #데이터등록하기      
    def insertMain(self):
        s = ScoreData()
        s.sname = input("이름 : ")
        s.kor = input("국어 : ")
        s.eng = input("영어 : ")
        s.mat = input("수학 : ")
        sql = """
            insert into tb_score(sname, kor, eng, mat)
            values(:sname, :kor, :eng, :mat)
        """
        params = [{"sname":s.sname, "kor":s.kor, "eng":s.eng, "mat":s.mat}]
        self.insert(sql, params)

    def insert(self, sql, params):
        with theEngine.begin() as conn:
            conn.execute(text(sql), params)

    #통계 
    #전체인원 : 45
    #수 : 12 
    #우 :
    #미
    #양
    #가 
    def statistic(self):
        sql="""
            select '전체' grade , count(*) cnt 
            from tb_score 
            union all 
            select grade, count(*) cnt
            from(
                select id, case  when (kor+eng+mat)/3>=90 then '수'
                                    when (kor+eng+mat)/3>=80 then '우' 
                                    when (kor+eng+mat)/3>=70 then '미'
                                    when (kor+eng+mat)/3>=60 then '양'
                                    else '가' 
                        end as grade 
                from tb_score
            ) A
            group by grade
            order by field(grade, '전체', '수', '우', '미', '양', '가')
        """
        temp={"전체":0, "수":0, "우":0, "미":0, "양":0, "가":0}
        with theEngine.connect() as conn:
            result = conn.execute(text(sql))
            for r in result.mappings().all():
                #print(r)
                temp[r["grade"]] = r["cnt"]

        print("전체",temp["전체"])
        print("수",temp["수"])
        print("우",temp["우"])
        print("미",temp["미"])
        print("양",temp["양"])
        print("가",temp["가"])
        


if __name__ == "__main__":
    sm = ScoreManager()
    #sm.insertMain()
    #sm.output()
    sm.statistic()
