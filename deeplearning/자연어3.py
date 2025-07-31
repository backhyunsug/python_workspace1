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

#download() #파일 다운받기 = 용량이 너무 커서 8192만큼씩 잘라서 저장하는 코드임 

#압축풀기 : 프로그램 호출 -> 프로세스 , tar 라이브러리가 있어야 한다 
def release():
    subprocess.run(["tar", "-xvzf", "aclImdb_v1.tar.gz"], shell=True) #tar 프로그램 가동하기 
    #tar.gz => linux에서는 파일을 여러개를 한번에 압축을 못함 tar라는 형식으로 압축할 모든 파일을 하나로 묶어서 패키지로 만든다음에 
    #          압축을 한다.  tar , gz가동  그래서 압축풀고 다시 패키지도 풀어야 한다. 
    #          tar  -xvzf 파일명   형태임         
    print("압축풀기 완료")

#release()

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

#labeling()

#데이터셋을 활용해서 디렉토리로부터 파일을 불러와서 벡터화를 진행한다 
import keras 
batch_size = 32 #한번에 읽어올 양 
train_ds = keras.utils.dataset_from_directory(
    "aclImdb/train", #디렉토리명 
    batch_size=batch_size
)

val_ds = keras.utils.dataset_from_directory(
    "aclImdb/val", #디렉토리명 
    batch_size=batch_size
)

test_ds = keras.utils.dataset_from_directory(
    "aclImdb/test", #디렉토리명 
    batch_size=batch_size
)

for inputs, targets in train_ds: #실제 읽어오는 데이터 확인 
    print("inputs.shape", inputs.shape)
    print("inputs.dtype", inputs.dtype)
    print("targets.shape", targets.shape)
    print("targets.dtype", targets.dtype)
    print("inputs[0]", inputs[0])
    print("targets[0]", targets[0])
    break #하나만 출력해보자 












