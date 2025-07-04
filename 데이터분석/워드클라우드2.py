#pip install simplejson

import pytagcloud 
import webbrowser 
from konlpy.tag import Okt
from collections import Counter #워드 카운트 단어 계수기 

file = open("./data/data1.txt", encoding="utf-8")
text = file.read() #텍스트파일 읽기 

#파일로부터 명사 추출 
okt = Okt() 
nouns = okt.nouns(text) #명사로 분해하기 

nounsCounter = Counter(nouns) #명사들 다 세서 (단어, 카운트) 형태로 데이터 전달
print(nounsCounter)

tag = nounsCounter.most_common(100) #모든 단어로 차트를 그리면 너무 정신없어서 
#빈도수를 기반으로 정렬한 다음 100개만 가져와서 차트를 그려라 

taglist = pytagcloud.make_tags(tag, maxsize=50)
print(taglist)
pytagcloud.create_tag_image(taglist, 
                            'wordcloud2.jpg',
                            size=(600,600),
                            fontname='Korean',
                            rectangular=False)
webbrowser.open('wordcloud2.jpg')

# import os 
# os.startfile('wordcloud2.jpg')

