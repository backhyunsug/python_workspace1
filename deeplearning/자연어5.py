import requests
import subprocess
import re
import string
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
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

#완전 원핫인코딩으로 해결했었는데 지금은 안함 

import tensorflow as tf 
embedding_dim = 600 
inputs = keras.Input(shape=(None,), dtype="int64")
#embeded = tf.one_hot(inputs, depth=max_tokens)#에러

from keras import models, layers
embedded = layers.Lambda(
    lambda x: tf.reshape(tf.one_hot(x, depth=max_tokens), (-1, tf.shape(x)[1], max_tokens)),  
    # Remove the extra dimension
    output_shape=(None, max_tokens)  # Specify the output shape
)(inputs) #원핫인코딩 -> 원핫인코딩 하고 나서 모델에 입력시킴 지금은 원핫인코딩 자체를 모델의 한단계로 추가 시킨다  
#시퀀스가 => 원핫인코딩으로 바꿔서 넣는 방법도 있고 지금처럼 함수 만들어서 전달해주면 알아서 처리가 된다. 
print(embedded.shape)

#양방향 RNN 을 가동시킴 
x = layers.Bidirectional( layers.LSTM(32))(embedded) 
x = layers.Dropout(0.5)(x) 
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10)
print("테스트셋 ", model.evaluate(int_test_ds))








