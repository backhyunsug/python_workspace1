#conda install selenium
#이전에는 드라이브파일을 다운받아서 연결하는 방식   

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options #브라우저를 처음 작동시킬때 옵션을 줄 수 있다 
from selenium.webdriver.common.by import By # 파싱도구 
from selenium.webdriver.common.keys import Keys # 키값제어 라이브러리 
import time 

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
driver.get("https://www.daum.net/") 

input_box = driver.find_element(By.NAME, "q") #내가 name속성을 이용해서 찾을 거다 
                                              #name이 q였음 
                                              #   
time.sleep(2) #프로세스가 cpu를 2초간 뺏긴다 
#input_box에 키값을 보냄 
input_box.send_keys("말복") #키보드 이벤트를 발생시킴
input_box.submit()  #서버로 정보를 전송 

#뉴스탭 
#daumGnb > div.tab_dynamic.tab_flex > ul > li:nth-child(2) > a
btnNews = driver.find_element(By.CSS_SELECTOR, 
                            "#daumGnb > div.tab_dynamic.tab_flex > ul > li:nth-child(2) > a")
btnNews.click() 

#dnsColl > div:nth-child(2) > div > div > a:nth-child(2)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(3)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(4)
#dnsColl > div:nth-child(2) > div > div > a:nth-child(5)

import random
import time 
from bs4 import BeautifulSoup

i=1 
while True: #무한루프 
    time.sleep( random.randint(1,12))
    print(f"{i} page .....")
    
    doc = BeautifulSoup(driver.page_source)
    ul = doc.find("ul", {"class":"c-list-basic"})
    liList = ul.find_all("li")
    for li in liList:
        print(li.text) 
        # title= li.find("span", {"class":"txt_info"})
        # print(title.text)
    
    #다음 페이지로 이동하기 
    i=i+1
    next = driver.find_element(By.CSS_SELECTOR, f"#dnsColl > div:nth-child(2) > div > div > a:nth-child({i})")
    if next == None or i>5:
        break 
    next.click()

#driver.quit() #종료 
#request -> url을 가져오면 전체 페이지가 한번에 만들어질 경우에는 상관없음 
#대부분 ajax 기반임, full stack  front => backend 




