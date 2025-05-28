from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# SQLAlchemy가 PyMySQL을 내부적으로 사용하며 pool 지원
engine = create_engine(
    "mysql+pymysql://root:1234@localhost:3306/mydb",
    pool_size=10,         # 최대 연결 수
    max_overflow=5,       # 초과 시 추가 연결 수
    pool_recycle=3600     # 재활용 시간
)

#1. 데이터가져오기
with engine.connect() as conn:
    sql = """
    select empno, ename, sal 
    from emp 
    """
    result = conn.execute(text(sql))
    for row in result.all():
        print(row)
    
    #dicttype으로
    result = conn.execute(text(sql))
    rows = result.mappings().all()
    for row in rows:
        print(dict(row))

#2.검색어를 전달할때 
ename = "조승연32" #키보드입력으로 변경 

with engine.connect() as conn:
    sql = """
        select empno, ename, sal 
        from emp 
        where ename=:name 
    """
    # :name
    result = conn.execute(text(sql), [{"name":ename}])
    temp = result.all()
    if len(temp) == 0:
        print("없음")
    else:
        for row in temp:
            print(row) 


#3.insert 
with engine.connect() as conn:
    sql = """
        select ifnull(max(empno), 0)+1 
        from emp 
    """
    result = conn.execute(text(sql)) #[()]
    empno = result.all()[0][0]
    sql = """
        insert into emp(empno, ename, sal)
        values(:empno, :ename, :sal)
    """
    conn.execute(text(sql), 
                 [{"empno":empno, "ename":"홍길동"+str(empno), 
                   "sal":9000}])
    conn.commit() #커밋 반드시


#3.insert  - 트랜잭션 처리 - 트랜잭션처리가 안되는경우 
# with engine.connect() as conn:
#     sql = """
#         select ifnull(max(id), 0)+1  from test1
#     """
#     result = conn.execute(text(sql))
#     id = result.all()[0][0]
#     sql ="""
#         insert into test1 values(:id, :field1)
#     """
#     conn.execute(text(sql), [{"id":id, "field1":"test"}])
#     conn.commit()

#     sql ="""
#         insert into test2 values(:id, :field1)
#     """
#     conn.execute(text(sql), [{"id":id, 
#                 "field1":"test12345678"}])
#     conn.commit()

#트랜잭션 처리가 필요할 경우에 
#ACID(atomic, consistancy, isolation, Durability ) 
with engine.begin() as conn:
    sql = """
        select ifnull(max(id), 0)+1  from test1
    """
    result = conn.execute(text(sql))
    id = result.all()[0][0]
    sql ="""
        insert into test1 values(:id, :field1)
    """
    conn.execute(text(sql), [{"id":id, "field1":"test"}])
    
    sql ="""
        insert into test2 values(:id, :field1)
    """
    conn.execute(text(sql), [{"id":id, 
                "field1":"test1234"}])
