from sklearn.datasets import load_files 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
import matplotlib.pyplot as plt 
import re #정규식 처리 하는 라이브러리 

#파일을 읽는다 - 구분자가 탭키다 
#csv파일이 , 콤마를 기준으로 데이터가 나눠져있는데 탭도 가능도 공백
#keep_default_na - NaN 값을 None으로 바꾼다
df_train = pd.read_csv("./data/aclImdb/ratings_train.txt", delimiter="\t", keep_default_na=False)
print( df_train.head() )
text_train, y_train = df_train["document"].values, df_train["label"].values
print( text_train[:3])
print( y_train[:3])

#text_traind을 벡터화하자 

from konlpy.tag import Okt  #현재 가장 많이 사용하는 형태소 분리 알고리즘 
okt = Okt() 
stop_words = ["아", "..", "?", "있는" ] #빼고 싶은단어들 리스트에 추가 
def okt_tokenizer(text):
    #특수문자를 제거하기 (숫자와 영어등을 제거 - 공백  )
    #정규식 패턴을 사용해서 문자바꿔치기 
    text = re.sub(r"[^\uAC00-\uD7A3\s]", "", text)
    temp = okt.morphs(text)#형태소 분리 
    #제거할거 있으면 제거시켜서 보내기, 불필요한 단어나 한글자는 삭제시키고 나머지만 
    temp = [word for word in temp if word not in stop_words and len(word)>=2]
    return temp 

for i in range(0, 10):
    print( okt_tokenizer(text_train[i] ))

#CountVectorizer의 tokenizer 매개변수에 우리가 토큰나이저 만들어서 주면된다. 
#한글 토큰나이저로 바꿔치기를 한다. 경고는 무시해도 된다. 
vect = CountVectorizer(tokenizer=okt_tokenizer).fit(text_train)

feature_names = vect.get_feature_names_out() 
print("특성의 개수 ", len(feature_names))
print(feature_names[:20])

X_train = vect.transform(text_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
print( model.score(X_train, y_train))










