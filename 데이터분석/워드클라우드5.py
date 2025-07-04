from wordcloud import WordCloud
import nltk #영어 자연어 처리 모듈 
from collections import Counter #단어개수
from konlpy.corpus import kolaw #한국 법률말뭉치
from konlpy.tag import  Okt  #한국 형태소 분석
import matplotlib.pyplot as plt #차트
import numpy as np 
from wordcloud import  ImageColorGenerator #이미지 컬러 체인지 
from PIL import Image #이미지처리용 이미지 파일 읽거나 쓰기를 담당한다. 
import random

#토큰이 문장에서 단어를 하나씩 분리 
#토큰나이저 - 문장에서 단어를 분리해내는 라이브러리 
#nltk.download('punkt') #토큰나이저 다운받는 코드 
nltk.download('punkt_tab') 
from nltk.tokenize import sent_tokenize #문장으로 쪼개주는 토큰나이저 
from nltk.tokenize import word_tokenize #단어로 쪼개주는 토큰나이저 
c = open("./data/alice.txt", "r").read() 
tokens_en = word_tokenize(c)
print(tokens_en)

data = Counter(tokens_en) #각 단어 개수를 센다 
print(data)

#앞에서 100개만 추려서 쓰자 
data = data.most_common(100) 
temp_data = dict(data)  #Counter => dict 타입으로 전환한다 
#이미지가 numpy형태의 배열로 전달되어야 한다. 
mask_image = np.array(Image.open("./image/alice_color.png"))

#generate_from_frequencies 
fontpath = "C:/Winodws/fonts/malgun.ttf"
wordcloud = WordCloud(font_path=fontpath,
                      mask=mask_image
                      ).generate_from_frequencies(temp_data)
plt.imshow(wordcloud, interpolation='bilinear') #이미지 좀 이뻐보이게 보간법 
plt.axis("off") #x,y축 안보이게 
plt.show()

# #from konlpy.tag import Okt  
# #plt 라이브러리가 한글 지원을 안한다. 한글을 쓰려면 폰트를 지정해야 한다. 
# #한국 법률 말뭉치 
# from konlpy.corpus import kolaw
# print(dir(kolaw))
# c = kolaw.open('constitution.txt').read() 
# print(c[:200])
# fontpath = "C:/Winodws/fonts/malgun.ttf"
# wordcloud = WordCloud(font_path=fontpath).generate(c)
# #파일로 저장 
# wordcloud.to_file("image1.png")
# # #이미지정보를 리턴 
# plt.imshow(wordcloud, interpolation='bilinear') #이미지 좀 이뻐보이게 보간법 
# plt.axis("off") #x,y축 안보이게 
# plt.show()
