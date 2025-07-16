from sklearn.datasets import load_files 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 

#파일불러오기 
reviews_train = load_files("./data/aclImdb/train")
print(reviews_train["DESCR"] )

text_train = reviews_train.data
y_train = reviews_train.target 

#bunch가 numpy 배열로 준다. => DataFrame 으로 바꿔서 쓰자 
df = pd.DataFrame( text_train, columns=["text"])
df["target"] = y_train 
df.to_csv("imdb.csv", encoding="utf-8-sig", index=False)

print(df.head())

#벡터화를 하자 => 일종의 비지도 학습 fit 학습을 하고 transform을 통해서 문장을 연산가능한 벡터로 만들어서 반환한다
vect  = CountVectorizer().fit(text_train) 
X_train = vect.transform(text_train)

#컬럼명을 만든다. 
feature_names = vect.get_feature_names_out() 
print("특성의 개수 ", len(feature_names))
print("특성 20개만 확인해보기", feature_names[:20])

#문자열앞에 b -> binary 약자, 쓸데없는것들 바꿔치기 <br /> 
text_train = [i.replace(b"<br />", b"") for i in text_train]

vect  = CountVectorizer().fit(text_train) 
X_train = vect.transform(text_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_train, y_train)
print(model.score(X_train, y_train))



















