import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from typing import Any, Dict, List, Tuple


class Database:
    url="mysql+pymysql://root:1234@localhost/mydb"
    def __init__(self, url=""): #생성자
        if url!="":
            self.url = url 
        
        self.engine = create_engine(self.url)
        self.connection = None 

    #with 구문을 썼을때 장점  with구문내애서 연결했다가 with구문 끝나면 자원이 해제 
    #예를들자면 db를 연결했다가 사용이 끝나고 해제를 안하면 에러는 안나지만 나중에 따로 해제가 될수도 있도있지만 
    #내자리에 스레기 잔뜩 있는데 언젠가는 청소해야 하지만 그상태로 계속 가니까 바로 정리하자 with구문이 끝나면 
    #바로 해제된다.  
    #연산자중복기능 with 를 사용하면 __enter__ 라는 함수가 자동 호출된다 이때 자원할당 
    #with가 끝날때 __exit__가 호출된다. 여기서 자원해제를 하자 
    def __enter__(self) -> 'Database':  #객체 반환 
        try:
            self.connection: Connection = self.engine.connect()
            #파이썬이 데이터타입이 없다. 시간이 지나면서 코드가 커지면 문제를 야기한다. 그래서 
            #변수를 써놓고 힌트   self.connection:타입힌트 
            #self.connection: Connection self.connection객체의 타입이 Connection이라는 의미임 
            print("DB연결성공")
            return self #자기자신을 반환한다 
        except sqlalchemy.exc.SQLAlchemyError as e:
            print("연결실패")
            raise #예외를 여기서 처리하지 않고 with 구문한테 던진다. 

    def __exit__(self, type, val, tb):
        if self.connection: #이미 닫힌게 아니면 
            self.connection.close() 
            print("데이터베이스 연결 종료")

    def execute(self, query:str,  args=None):
        #insert, update, delete 담당 
        with self.connection.begin():#트랜잭션
            self.connection.execute(text(query), args) 

    #결과 하나 반환 scalar쿼리실행 결과를 반환 
    def executeOne(self, query:str,  args=None):
        result =  self.connection.execute(text(query), args) 
        row = result.fetchone() 
        return dict(row._mapping) if row else None  #tuple => dict타입으로 바꿔서 반환하기 
    
    #전체 데이터 가져오기 
    def executeAll(self, query:str,  args=None):
        result =  self.connection.execute(text(query), args)  
        return [dict(row._mapping) for row in result.fetchall()]   


