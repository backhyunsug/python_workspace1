#conda install selenium
#이전에는 드라이브파일을 다운받아서 연결하는 방식   
#캡처하기
  
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options #브라우저를 처음 작동시킬때 옵션을 줄 수 있다 
from selenium.webdriver.common.by import By # 파싱도구 
from selenium.webdriver.common.keys import Keys # 키값제어 라이브러리 
import time 
from bs4 import BeautifulSoup 

#1.Options객체를 만들어서 크롬드라이버의 기본 설정을 관리한다 
chrome_options = Options() 
#chrome_options.add_argument("--window-size=1920,1000") #윈도우 크기 지정 
chrome_options.headless = False      #옵션은 gpt한테 물어보기 
#브라우저가 켜지고 작업완료를 하면 자동으로 닫힌다. - 그걸 막고 싶을때 사용하는 옵션
chrome_options.add_experimental_option("detach", True)
#detach-분리 attech-접속 

#2. 크롬드라이버를 실행하고 옵션을 적용한다 
driver = webdriver.Chrome(options = chrome_options)
#driver객체가 브라우저를 객체임 
driver.implicitly_wait(3)   #페이지가 열릴때까지 3초쯤 기다려라 
driver.get("http://www.python.org") 
#assert 디버깅 툴
assert "Python" in  driver.title  
input_box = driver.find_element(By.NAME, "q") #내가 name속성을 이용해서 찾을 거다 
                                              #name이 q였음 
                                              #   
time.sleep(2) #프로세스가 cpu를 2초간 뺏긴다 
#input_box에 키값을 보냄 
input_box.send_keys("python") #키보드 이벤트를 발생시킴
time.sleep(3)

#1. input_box.submit()  #서버로 정보를 전송
#웹개발자들이 커서가 있는곳에서 엔터키를 누르면 서버로 전송한다.  
#2. input_box.send_keys(Keys.RETURN)  #Keys 엔터키를 누른다. 
#내부적으로 엔터키를 누르면 => submit함수가 호출되게 해 놓았을때 동작함 
#3.서브밋트 버튼을 찾아서 이벤트를 발생시킨다 
button = driver.find_element( By.NAME, "submit")
button.click() #버튼눌림 이벤트 => click이벤트핸들러가 submit()를 호출함 

#셀레니움도 문서를 불러올 수 있다. page_source : html페이지 
print(driver.current_url) #현재 문서 url 
print(driver.title) #현재 문서 제목
#driver.page_source #현재 문서  

driver.set_window_size(400,400) 
driver.save_screenshot("python_event.png") #화면캠쳐후 파일저장

#다음페이지로 이동하기 
#  #content > div > section > form > div > a:nth-child(2)
#  content > div > section > form > div > a

driver.set_window_size(800,600)

print("************************************************")
next = driver.find_element(By.CSS_SELECTOR, "#content > div > section > form > div > a:nth-child(1)")
print(next)
next.click()

time.sleep(5)
# //*[@id="content"]/div/section/form/div/a[2]

driver.quit() 




