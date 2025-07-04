import matplotlib.pyplot as plt
from wordcloud import WordCloud 

file = open("./data/alice.txt")
text = file.read() 
print(text)
wordcloud = WordCloud().generate(text)
#이미지정보를 리턴 
plt.imshow(wordcloud, interpolation='bilinear') #이미지 좀 이뻐보이게 보간법 
plt.axis("off") #x,y축 안보이게 
plt.show()
