#pip install pymongo
import pymongo
from pymongo import MongoClient 

# db 연동- 목록 가져오기 
from pymongo import MongoClient
client = MongoClient("mongodb://test:1234@127.0.0.1:27017/")
db = client.mydb 
rows = db.person.find()
for row in rows :
    print(row)
    #print(row['_id'], row['name'], row['gender'])

    

"""
콘솔에서는 복사 ctrl-enter키 
붙여넣기 마우스 오른쪽 버튼 누르기
"""
