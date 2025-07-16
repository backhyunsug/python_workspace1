#텍스트분석 : +, - , *, / 연산이 가능하게 바꿔줘야 한다. 
#sklearn이  CounterVectorizer 라는 클래스를 제공한다. => RNN(순환신경망)을 주로 사용한다. 
bards_words = ["I like star", 
               "red star, blue star",
                "I like dog" ]

from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer() #객체 만들고 
#1. 사전을 만들고 - 학습해서 
vect.fit(bards_words)

print("어휘사전 크기 : ", len(vect.vocabulary_) )
print("어휘사전 내용 : ", vect.vocabulary_ )

#2.사전을 바탕으로 벡터화 
bag_of_words = vect.transform(bards_words)
print(bag_of_words  )
print(bag_of_words.toarray())

#로이터 뉴스, IMDB 영화평론 사이트 
#1.토큰화 -> 토큰을 나눈다. 단어로 나눈다. 바꿔서 처리가 가능하다 
#2.어휘사전 구축 -> 단어한테 고유 숫자를 준다. I like star => I :0  like:2  star:2 
#                 어휘사전이 충분히 커야 한다 
#3. 단어빈도수계산  like:10, star:20, red:4 ...........  
#4. 문서단어 행렬 => 단어 빈도수를 행렬로 표현한다. 0이 엄청많다. 희소행렬이라고 한다  
