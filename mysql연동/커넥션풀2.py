#pip install dbutils 
#설치경로 : C:\Users\littl\AppData\Roaming\Python\Python38\site-packages\dbutils
from dbutils.pooled_db import PooledDB
import pymysql

# PooledDB를 이용한 커넥션 풀 구성
pool = PooledDB(
    creator=pymysql,
    maxconnections=10, 
    mincached=2,
    blocking=True,
    host='localhost',
    user='root',
    password='1234',
    database='mydb',
    charset='utf8mb4' #문자셋 , mysql버전낮을때
)

# 커넥션 얻기
conn = pool.connection()
cursor = conn.cursor()
cursor.execute("SELECT * FROM tb_score")
print(cursor.fetchall())
cursor.close()
conn.close()