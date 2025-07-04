import matplotlib.pyplot as plt
from wordcloud import WordCloud 
#from konlpy.tag import Okt  
#plt 라이브러리가 한글 지원을 안한다. 한글을 쓰려면 폰트를 지정해야 한다. 
#한국 법률 말뭉치 
from konlpy.corpus import kolaw
print(dir(kolaw))
c = kolaw.open('constitution.txt').read() 
print(c[:200])
fontpath = "C:/Winodws/fonts/malgun.ttf"
wordcloud = WordCloud(font_path=fontpath).generate(c)
#파일로 저장 
wordcloud.to_file("image1.png")
# #이미지정보를 리턴 
plt.imshow(wordcloud, interpolation='bilinear') #이미지 좀 이뻐보이게 보간법 
plt.axis("off") #x,y축 안보이게 
plt.show()
