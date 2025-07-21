#뷰티플스프를 이용해서 html파싱 
#로컬에 있는 문서를 읽어서 파싱하는것만 진행한다. 
html = open("./html/test1.html", encoding="utf-8")
doc = html.read() #파일의 크기가 작아서 파일을 한번에 다 읽는다 
html.close() 

from bs4 import BeautifulSoup 

soup = BeautifulSoup(doc, "html.parser") #html -> DOM구조 
#find("태그", "속성") - 첫번째것만       
#findAll("태그", "속성") - 언제나 list형태, 인덱싱이나 for문으로 접근하면 된다. 

#태그 객체 가져오기 
title_tag = soup.find("title")  #<title>~</title>
print(title_tag)
print("내용만 ", title_tag.text)

#h1태그 가져오기 
h1 = soup.find("h1")#첫번째 거 하나만 가져온다 
print(h1.text)

#h1태그 전체 가져오기 
h1List = soup.find_all("h1")
for h1 in h1List:
    print(h1.text)

#css selector  id하고 class이용하기 
print("---- 태그와 id로 접근하기 ----")
hList = soup.find("h1", id="title1")       #같은문법 
hList = soup.find("h1", {"id":"title1"})
for h1 in hList:
    print(h1.text)

print("---- 태그와 class로 접근하기 ----")
hList = soup.find("h1", __class="title_red")       #같은문법 
hList = soup.find("h1", {"class":"title_red"})
for h1 in hList:
    print(h1.text)

print("------ ul 태그 가져오기 --------")
#1.ul태그 가져와서 이 태그로부터 li태그 리스트를 가져온다
ul = soup.find("ul", {"class":"coffee"})
print(ul) #못찾으면 None값이 온다 
liList = ul.find_all("li")
for li in liList:
    print(li.text)

print("----- 테이블태그가져오기 ----------")
table = soup.find("table", id="productList")
trList = table.find_all("tr")
for tr in trList:
    tdList = tr.find_all("td")
    for td in tdList:
        print(td.text, end=" ")
    print()






