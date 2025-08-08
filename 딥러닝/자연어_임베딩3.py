#사전학습된 임베딩 계층을 이용한다. Glove 
#설치해야 할 라이브러리 gensim 
#conda install gensim 
"""
Word2Vec 다운받기
Direct Link (S3 AWS):
 https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

Hugging Face: 최근에는 Hugging Face datasets나 models에서도 찾아볼 수 있습니다.
 예를 들어, NathaNn1111/word2vec-google-news-negative-300-bin와 같은 저장소에서 다운로드 링크를 제공합니다.

Kaggle: Kaggle에도 이 데이터셋이 업로드되어 있습니다.
 GoogleNews-vectors-negative300.bin.gz를 검색하면 찾을 수 있습니다.
 https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

"""
import requests
import subprocess
import re
import string
import tensorflow as tf
from keras.layers import TextVectorization
import os, pathlib, shutil, random

#데이터 다운로드 
def download():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_name = "aclImdb_v1.tar.gz"

    response = requests.get(url, stream=True)  # 스트리밍 방식으로 다운로드
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  # 8KB씩 다운로드
            file.write(chunk)

    print("Download complete!")


#압축풀기 : 프로그램 호출 -> 프로세스 , tar 라이브러리가 있어야 한다 
def release():
    subprocess.run(["tar", "-xvzf", "aclImdb_v1.tar.gz"], shell=True) #tar 프로그램 가동하기 
    #tar.gz => linux에서는 파일을 여러개를 한번에 압축을 못함 tar라는 형식으로 압축할 모든 파일을 하나로 묶어서 패키지로 만든다음에 
    #          압축을 한다.  tar , gz가동  그래서 압축풀고 다시 패키지도 풀어야 한다. 
    #          tar  -xvzf 파일명   형태임         
    print("압축풀기 완료")


#train => train과 validation으로 나눠야 한다. , train 폴더에 있는 unsup 폴더는 직접 지워냐 한다. 
#라벨이 2개 여야 한다. 

#라벨링 
def labeling(): 
    base_dir = pathlib.Path("aclImdb") 
    val_dir = base_dir/"val"   # pathlib 객체에  / "디렉토리" => 결과가 문자열이 아니다 
    train_dir = base_dir/"train"

    for category in ("neg", "pos"):
        os.makedirs(val_dir/category)  #디렉토리를 만들고 
        files = os.listdir(train_dir/category) #해당 카테고리의 파일 목록을 모두 가져온다 
        random.Random(1337).shuffle(files) #파일을 랜덤하게 섞어서 복사하려고 파일 목록을 모두 섞는다 
        num_val_samples = int(0.2 * len(files)) 
        val_files = files[-num_val_samples:] #20%만 val폴더로 이동한다 
        for fname in val_files:
            shutil.move(train_dir/category/fname, val_dir/category/fname )    

#download() #파일 다운받기 = 용량이 너무 커서 8192만큼씩 잘라서 저장하는 코드임 
#release()
#labeling()

#데이터셋을 활용해서 디렉토리로부터 파일을 불러와서 벡터화를 진행한다 
import keras 
batch_size = 32 #한번에 읽어올 양 
train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", #디렉토리명 
    batch_size=batch_size
)

val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", #디렉토리명 
    batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", #디렉토리명 
    batch_size=batch_size
)
#데이터셋은 알아서  inputs, targets 을 반복해서 갖고 온다. 우리한테 필요한거는 inputs만이다
for inputs, targets in train_ds: #실제 읽어오는 데이터 확인 
    print("inputs.shape", inputs.shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[:3])
    print("targets[0]", targets[:3])
    break #하나만 출력해보자 
#0이 부정 1이 긍정 -> 폴더명을 정렬해서 0,1,2 이런식으로 라벨링을 한다 neg -0 pos-1 

#시퀀스를 만들어야 한다 
max_length = 600  #한 평론에서 사용하는 단어는 최대 길이를 600개라고 보자  
max_tokens = 20000 #자주 사용하는 단어 20000 개만 쓰겠다 

text_vectorization = TextVectorization( 
    max_tokens = max_tokens,
    output_mode = "int", #임베딩 층을 사요하려면 반드시 int여야 한다
    output_sequence_length = max_length  
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds) #어휘사전을 만들어야 한다 

int_train_ds = train_ds.map( lambda x,y:(text_vectorization(x), y), num_parallel_calls=1 )
int_val_ds = val_ds.map( lambda x,y:(text_vectorization(x), y), num_parallel_calls=1 )
int_test_ds = test_ds.map( lambda x,y:(text_vectorization(x), y), num_parallel_calls=1 )

#내부구조 살짝 보기 
for item in int_train_ds:
    print(item)
    break 

#Word23Vec 파일 불러오기 

from gensim.models import KeyedVectors 
filename = "./data/GoogleNews-vectors-negative300.bin"

try:
    word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    #파일이 txt형태가 있고 binary 형태가 있다. binary형태를 읽겠다
    embedding_dim = word2vec_model.vector_size 
except FileNotFoundError:
    print(filename + "을 찾을 수 없습니다")
    exit()#프로그램 종료
except Exception as e :
    print("에러발생 " + e)
    exit()#프로그램 종료
print("파일 로딩 성공")

#임베딩층 - 내부적으로는 연산을 해서 단어와 단어사이의 관계를 계산해서 밀집벡터를 만든다. 
#원핫인코딩 - 메모리를 너무 많이 차지함   최대한 한문장을 표현하는데 만일 최대 20000 단어까지 처리한다면 
#한문장당 20000개가 필요 , 희소행렬 요소가 거의다 0인데 그중 몇개가 값이 있을때, 학습시 속도가 엄청 느리다 
#임베딩층을 사용한다. => 단어와 단어사이의 거리를 재는 방식인데, 비슷 비슷한 단어가 같은 문장에 나타난다고 전제 하고 문제를 해결한다 
#틀릴수 있다.
#케라스가 Embedding 레이어를 제공한다. 이 레이어는  반드시 정수 인덱스를 받아야 한다. 시퀀스를 받아서 밀집벡터를 만든다 
#알고리즘 공개안함. 파이썬은 소스 공개 안할 방법이 없다.   ProgramData\anaconda3\envs\가상환경\libs\site-packages    

# #사전학습된 임베딩 데이터를 불러온다
# import numpy as np 
# path_to_glove_file = "./data/glove.6B.100d.txt"
# #임베딩 데이터, 단어별 각 단어와의 거리가 벡터로 저장되어 있음  파일명의 100이 출력 벡터의 크기이다 
# embeddings_index = {}
# with open(path_to_glove_file, encoding="utf-8") as f:
#     for line in f: #한 라인씩 읽는다 
#         #단어, 단어들간의 벡터 구조로 되어 있다  예) the 0.0012 000172  ...... 
#         word, coefs = line.split(maxsplit=1)  
#         coefs = np.fromstring(coefs, "f", sep=" ") #나머지 벡터들을 numpy배열로 전환
#         embeddings_index[word]=coefs
#         #print( embeddings_index)
         
# # print("개수 ", len(embeddings_index))
# # #print(embeddings_index) 
# # #{"the":[ , , , , ,]}

#우리데이터와 연동을 해야 한다 
import numpy as np
vocabulary = text_vectorization.get_vocabulary() #우리 어휘사전 가져오기
#{단어:인덱스} 형태의 딕셔너리르 만들어야 한다 
#{"", "[UNK]", "write", "love", "make",...........} vocaburary
#{0,1,2,3,4,5,6,..................}
#zip ("",0) ("[UNK]",1) ("write", 2 )............
#{"":0, "[UNK]":1, "write":2, ,,,,,,,}

word_index = dict(zip(vocabulary, range(len(vocabulary)))) 
embedding_matrix = np.zeros((max_tokens, embedding_dim))  # 20000 * 100 배열을 잡고 0으로 채운다 
#케라스  Embedding 레이어에 초기값으로 쓰여야 한다.  
#embedding_matrix 를 우리가 embedding_index 정보로 채워야 한다. 

#word_index 는 {단어:인덱스} 형태임 
for word, i in word_index.items(): 
    #단어와 인덱스를 가져온다 
    if i <max_tokens: #혹시나 20000개를 넘어가는 토큰이 있을까봐 오류처리
        try:
            embedding_vector = word2vec_model[word] #단어에 해당하는 벡터들 이동
            embedding_matrix[i] = embedding_vector
        except KeyError:
            pass  #건너뛰기만  
    
print( embedding_matrix[:10])


from keras import models, layers
import tensorflow as tf 
import keras 

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(
    input_dim = max_tokens, 
    output_dim=embedding_dim, 
    embeddings_initializer=keras.initializers.Constant(embedding_matrix), ####반드시   
    #사전학습된 층에 의해 바꿔치기가 이뤄져야 한다 
    trainable=False,  #임베딩가중치를 훈련중에 업데이트할거냐? 사전학습돈 임베딩층을 사용할때는 False로 지정해야 한다 
    mask_zero=True 
    )(inputs)
 
#미리 학습된 임베딩층으로 바꿀 수가 있다
#입력벡터크기는 20000, 출력벡터는 256(특별한 의미 없이 마음대로)의 크기를 갖는다  
print(embedded.shape)
# #양방향 RNN 을 가동시킴 
x = layers.Bidirectional( layers.LSTM(32))(embedded) 
x = layers.Dropout(0.5)(x) 
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(int_train_ds, validation_data=int_val_ds, epochs=15)
print("테스트셋 ", model.evaluate(int_test_ds))












