#벡터화 - 자연어 처리가 TextVectorization을 수정해서 우리 코드를 넣는다 
from konlpy.tag import Okt 
import tensorflow as tf 
import re 
from tensorflow.keras.layers import TextVectorization 

#1. 한글 형태소 분석 모듈 객체 생성 
okt = Okt() 
#2.어휘사전을 만들 텍스트 준비 
texts = [ 
    "나는 ai 챗봇을 사용합니다. ",
    "자연어 처리는 어렵습니다", 
    "챗봇과 대화하는것이 즐겁습니다",
    "Ai로 인해 사람들이 일자리가 사라지고 있습니다."
]

def clean_text(text):
    text = text.lower() #대문자를 -> 소문자로 바꿔서 처리하자 
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]", "", text)
    return text

#토큰나누는 함수 
def custom_standardization_fn(text):
    #쓸데없는 문자들 제거 
    text = clean_text(text)
    return " ".join(okt.morphs(text)) #토큰화 하고 다시 합치기 - 토큰과 토큰은 공백으로 구분한다 

#TokenVercorization 객체 만들기 
vectorizer = TextVectorization(
    max_tokens = 1000,   #TextVactorization은 단어들을 토큰으로 분해해서 사용 숫자대로 정리한다. 앞에서 자주 쓰는 단어 1000
                         #개로 어휘사전을 만들겠다는 의미이다 어휘사전에 사용될 단어가 많으면 크기를 늘리면 된다. 
    output_mode = "int", #시퀀스로 만들라는 의미임 
    output_sequence_length = 20, #한 문장에 쓰일 단어의 길이는 최대 20개로 한다 
    standardize=None             #외부에서 데이터를 변경해서 보내줄 경우에 별도의 정규화함수를 사용하지 않는다. 
                                 #None값을 줘야 한다   
)

#Okt로 해서 변환한 데이터로 어휘사전을 만들어야 한다 
# OTK로 변환된 데이터 학습
data = [custom_standardization_fn(text) for text in texts] 
#데이터셋을 사용해서 파일을 읽어서 처리하려면 Tensorflow가 제공하는 데이터타입으로 바꿔서 전달되어서 
#파이썬에서 쓸 수 있게 바꿔서 Okt적용하고 나서 Tensor타입으로 바꿔서 넣어야 한다. 
print(data)
vectorizer.adapt(data) #어휘사전을 만든다. <== 데이타셋 자체가 커야 한다. 

# adapt 이후 어휘 사전 확인
vocabulary = vectorizer.get_vocabulary() #어휘사전 가져오기 
print("--- 어휘 사전 (get_vocabulary) ---")
print(vocabulary)
print("-" * 30)

# 어휘 사전을 단어:숫자 형태의 딕셔너리로 변환
word_to_index = {word: index for index, word in enumerate(vocabulary)}

print("--- 단어:숫자 형태의 어휘 사전 (딕셔너리) ---")
print(word_to_index)

#벡터화 결과 확인 
vectorized_text = vectorizer( [custom_standardization_fn("챗봇과 이야기를 합니다"), 
                               custom_standardization_fn("챗봇보다 재미나이가 더 좋아요"), 
                               custom_standardization_fn("자연어 처리는 어려워요"), 
                               ])
print(vectorized_text)

