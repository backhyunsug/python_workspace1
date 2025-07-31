import re #정규식 처리 
import string 
import tensorflow as tf 
from keras.layers import TextVectorization 

def custom_standardization_fn(text):
    lower_text = tf.strings.lower(text)
    return tf.strings.regex_replace( lower_text, f"[{re.escape(string.punctuation)}]", "") 

def custom_split_fn(text):
    return tf.strings.split(text)

#객체 만들때 파라미터 값들을 지정하면 된다. 
text_vectorization = TextVectorization(
    output_mode = "int", #출력이 시퀀스임 
    standardize =custom_standardization_fn,  #표준화에 사용할 함수를 전달할 수 있다.
    split=custom_split_fn  #토큰화에 사용할 함수를 전달할 수 있다
)

dataset=[
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms",
    "Dog is pretty"
]

text_vectorization.adapt( dataset) #학습시킬 데이터가 았으면 adapt에 전달하면 된다. 

vocaburary = text_vectorization.get_vocabulary()
#알아서 단어 빈도수로 정리됨  
print(vocaburary )

#인코딩 
text = "I wruite, rewrite, and still rewrite agian"
encoded = text_vectorization(text)
print(encoded)

#디코딩 
decoded_voca = dict(enumerate(vocaburary))
print(decoded_voca )

decoded_sen = " ".join(decoded_voca[int(i)] for i in encoded)
print(decoded_sen)