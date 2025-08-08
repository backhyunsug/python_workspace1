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

#데이터셋이  문장과 라벨로 연결되어, 라벨은 이미 정수화되어서 오니까 이거 버리고, 문장만 벡터화를 해야 한다 
text_vectorization  = TextVectorization(
    max_tokens =20000, #자주 사용하는 단어 20000개만 지정함, 
    output_mode = "multi_hot" #벡터로, 각 리뷰마다 20000개의 요소를 갖는 배열이 만들어진다. 
    #배열에서 문장중에 단어가 있는곳은 1 아니면 0으로 바꿔온다 
    #멀티핫 인코딩 또는 BoW(Bag Of Word, 단어가방) 방식  
)

#가져온 train_ds 에서 문장만 필요하다 
text_only_train_ds = train_ds.map( lambda x, y:x)   #데이터셋으로부터 문장만 추출 
text_vectorization.adapt(text_only_train_ds) #어휘사전 만들기 

####  각 데이터별로 이 작업을 진행해야 한다. 
#유니그램 - 단어를 1개씩 가져오는 방식 

#멀티프로세싱 , 한번에 cpu코더 4개를 사용해서 작업을 수행한다. 
binary_1gram_train_ds = train_ds.map( lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_val_ds   = val_ds.map(   lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_test_ds  = test_ds.map(  lambda x, y : (text_vectorization(x), y), num_parallel_calls=4)

print("벡터화 후 ----------------------------------")
for inputs, targets in binary_1gram_train_ds: #실제 읽어오는 데이터 확인 
    print("inputs.shape", inputs.shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[:3])
    print("targets[0]", targets[:3])
    break #하나만 출력해보자 


#모델 만들어서 반환하는 함수
from keras import layers, models 
def getModel(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,)) #입력층 만들기 
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x) 
    model = keras.Model(inputs, outputs) 
    model.compile( optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model 

model = getModel() 
model.summary() 
callbacks = [ 
    keras.callbacks.ModelCheckpoint("binary_1gram.keras", save_best_only=True) 
]
#적절한 시점에서 파일 저장하기 

#cache : 데이터셋을 메모리에 캐싱한다. 첫번재 에포크에서 전처리를 한번만 하고 더이상 하지 않고 재사용을 한다 
#메모리에 들어갈 만큼 작은 데이터셋일때만 가능하다 
model.fit(binary_1gram_train_ds.cache(), 
          validation_data=binary_1gram_val_ds, 
          epochs=10, 
          callbacks=callbacks)

model = models.load_model("binary_1gram.keras")  #학습한 내용 읽어보기 
print("테스트셋 정확도 ", model.evaluate( binary_1gram_test_ds))
 
























