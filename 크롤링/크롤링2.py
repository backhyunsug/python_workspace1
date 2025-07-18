#pip install bs4 
import requests
from bs4 import BeautifulSoup 
response = requests.get("http://www.pythonscraping.com/exercises/exercise1.html") 
print("응답코드 ", response.status_code) #200이면 성공 , 404 페이지없음 403 권한없음 500 서버에러 
if response.status_code==200:
    #response.text   => 정보를 받아올때 문자열로 받아온다 
    #response.content => binary 모드로 가져옴, 이미지나 동영상이나 처리할때
    print(response.text)
    #파싱작업하기 
    bs = BeautifulSoup(response.text, "html.parser") 
    #이 상태로 이미 파싱 끝나서 내부에 DOM구조를 갖고 있음 
    print(bs.title.text) #<title>태그</title>
    print(bs.h1.text)
else:
    print("error")

    
