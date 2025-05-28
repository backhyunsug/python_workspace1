#conda activate myenv1 
#pip install sqlalchemy 
#버전 2.0이상 
#C:\Users\littl\AppData\Roaming\Python\Python38\site-packages\sqlalchemy
#https://soogoonsoogoonpythonists.github.io/sqlalchemy-for-pythonist/tutorial/1.%20%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC%20%EA%B0%9C%EC%9A%94.html#%E1%84%8C%E1%85%A6%E1%84%80%E1%85%A9%E1%86%BC%E1%84%83%E1%85%AC%E1%84%82%E1%85%B3%E1%86%AB-%E1%84%80%E1%85%A5%E1%86%BA

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# SQLAlchemy가 PyMySQL을 내부적으로 사용하며 pool 지원
engine = create_engine(
    "mysql+pymysql://root:1234@localhost/mydb",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)

try:
    conn = engine.connect() #연결객체를 얻는다 
    print("데이터베이스 연결 성공")
except SQLAlchemyError as e:
    print("데이터베이스 연결 실패:", e)


#2.0 이전 버전 conn.execute("쿼리")
result = conn.execute(text("SELECT * FROM emp"))
#tuple로 출력한다 
# for row in result:
#     print(row)

#dicttype 으로 출력
rows = result.mappings().all()
for row in rows:
        print(dict(row))
conn.close()

#데이터추가하기 - 파라미터 처리방식
conn = engine.connect() 
sql = text("""
    insert into emp (empno, ename, sal )
    values(:empno, :ename, :sal) 
""")
conn.execute( sql, [{"empno":10001, 
                     "ename":"우즈2", 
                     "sal":8000}] )
conn.commit() 
conn.close() 

"""
orm - object - relation - mapping
      객체지향언어 - 관계형디비- 연결하는 프로그램 기법 
      django(장고, 파이썬 웹프레임워크),
      jpa - java 
      파이썬에서 SqlAlchemy 가 지원한다 
      전체테이블이 10개 미만일때 
      테이블 3개이상 조인시 속도문제, 테이블간의 구조가 복잡해질
      수록 orm만으로는 처리가 되지 않는다  
SPA - single page application, 인스타그램, 페이스북 등의
    개발기법, 웹페이지를 하나만 만들어서 페이지 체인지를 
    하는방식으로 화면이동이 훨씬 매끄럽다 무한스크롤등을 지원한다
    react(압도적), vuejs(우리나라 금융권), angular, polymer(유투부)   
    인스타그램 개발하다가 만들은게 react라이브러리 
    spa 방식 웹개발시 orm방식을 많이 사용한다  

sqlite - 본래의미의 디비는 아니고 사실은 파일이고 
        다만 쿼리형식으로 읽고 쓰기는 가능하나 
        네트워크가 안됨 , 휴대폰의 전화번호부 
"""