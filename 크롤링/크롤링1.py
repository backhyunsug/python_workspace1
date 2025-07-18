#웹스크래핑, 크롤링, 웹사이트마다 데이터를 갖고 오는 방식이 다르다 
#계속 웹사이트가 업그레이드가 되어서, 너무 많이들하니까 막는 기술도 있긴
#리캡챠 

#urllib 옛날 ==>  requests 라는 모듈이 있음 
#1. requests 모듈에 get방식, post방식으로 웹서버랑 접속해서 문서를 가져올 수 있다 
#문서, 이미지, 파일 ..............
#서버쪽에서 보내는 응답을 받아온다. html(일반웹서버), json형태임(restpul api서버)
#html -> 이 문서를 분석해서 데이터만 추출(파싱이라고 한다)
#html문서로부터 데이터를 파싱하는 알고리즘 BeautifulSoup - 설치를 해야한다 
#json라이브러리 

#셀레니움 - 크롬을 만들다가 디버깅용 툴을 만들음, 사용이 어려움 웬만하면 된다.
#        - 이벤트나 자바스크립트 호출이 가능하다.   

#1단계 : 파이썬에서 requests 모듈을 사용해서 문서를 불러오자 
#웹클라이언트  ========= request(요청)   ==================> 웹서버
#            <======== response(응답)  ================== 
  
import requests 
response = requests.get("http://www.pythonscraping.com/exercises/exercise1.html") 
print("응답코드 ", response.status_code) #200이면 성공 , 404 페이지없음 403 권한없음 500 서버에러 
if response.status_code==200:
    #response.text   => 정보를 받아올때 문자열로 받아온다 
    #response.content => binary 모드로 가져옴, 이미지나 동영상이나 처리할때
    print(response.text)
else:
    print("error")

    
