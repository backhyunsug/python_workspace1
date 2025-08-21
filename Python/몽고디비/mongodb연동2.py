import pymongo
from pymongo import MongoClient 

# db 연동- 목록 가져오기 
from pymongo import MongoClient
conn = MongoClient("mongodb://test:1234@127.0.0.1:27017/")
db = conn.mydb 
rows = db.person.find()
for row in rows :
    print(row)
    
# collection 생성
guestbook = db.guestbook # 새로운 컬랙션 생성하기 

#document 생성 : {'key':'value'}
guestbook.insert_one({'id':1, 'title':'제목1', 'contents':'내용1', 'writer':'홍길동', 
'wdate':'2019-03-15', 'age':23}); 
guestbook.insert_one({'id':2, 'title':'제목2', 'contents':'내용2', 'writer':'임꺽정', 
'wdate':'2019-03-16', 'age':33}); 
guestbook.insert_one({'id':3, 'title':'제목3', 'contents':'내용3', 'writer':'장길산', 
'wdate':'2019-03-17', 'age':42}); 
guestbook.insert_one({'id':4, 'title':'제목4', 'contents':'내용4', 'writer':'홍경래',
 'wdate':'2019-03-18', 'age':53}); 
guestbook.insert_one({'id':5, 'title':'제목5', 'contents':'내용5', 'writer':'장승업', 
'wdate':'2019-04-12', 'age':34}); 

rows = db.guestbook.find()
for row in rows :
    print(row)

#문서 삭제 : db.collection.remove() -> collect.remove()
guestbook.delete_one({'id' : '1'})
#전체삭제 : guestbook.remove({})

#문서 수정 : db.collection.update_one()
guestbook.update_one( {'id':2}, {'$set':{'title':'제목을 수정합니다'}}, True)
#조건, 수정할거, 똑같은거 모두 수정할건지 true/false

# 전체 문서 조회
rows = db.guestbook.find()
for row in rows :
    print(row)
    
# 조건 검색
print('조건 검색')
result2 = guestbook.find({ 'age':{'$gte':30} }) #크거나 같다
for r in result2 :
    print(r)

#컬렉션 제거
guestbook.drop()

