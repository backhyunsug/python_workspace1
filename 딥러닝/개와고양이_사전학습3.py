#개와고양이_사전학습1.py

#사전학습된 모델을 가져다 사용하기 (특성추출-투스테이지, 인라인(좀더 핫함), 미세조정)
#1.특성추출하기 - CNN하고 완전피드포워딩     합성곱신경망(CNN) + 완전피드포워딩 

#2.이미 학습된 모델을 불러와서 CNN파트랑 완전연결망을 쪼개서 
#  CNN으로 부터 특성을 추출한 다음에 완전연결망한테 보내서 다시 학습(분류학습)을 한다
#  CNN이 시간이 많이 걸린다. => CNN재활용을 하면 학습시간도 적게 걸리고, 예측률도 더 높아진다. 
# 이미 수십만장의 사진을 가지고 학습한 모델을 갖다 쓴다 
#     장점) 데이터셋이 적을 경우(1000장)
#          이미 학습된 모델을 사용함으로써 학습시간을 줄여준다
#          컴퓨터 자원이 작아도 학습이 가능하다. 
# VGG19, ResNet, MobileLet 등 이미지셋 모델들이 있다. 
# https://hwanny-yy.tistory.com/11
"""
1. 투스테이지(개와고양이_사전학습1.py) 
CNN동결   VGG19의 특징을 미리 계산하고 numpy배열로 바꾼다 저장된 특성으로 분류학습을 다시한다 
장점 : 훈련속도가 빠르다 
단점 : 메모리를 많이 차지한다 
      학습하고자하는 데이터셋이 커지면 힘들다. 데이터증강 적용방식이 내 데이터가 아니라 추출한 특성에 적용된다. 

2. 인라인(개와고양이_사전학습2.py) 
VGG19특징 추출부분을 전체 모델안에 포함시킨다. 분류학습을 한다
장점 : 원본이미지에 데이터 증강이 바로 적용된다. 모델의 과대적합을 막을 수 있다. 
속도는 투스테이지보다 느리다. 
사람들이 많이 쓰는 방법이다.  

"""
import gdown #케라스만든사람들이 케라스에 있는 데이터셋 업어오기위해 사용함 
#gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')
#안막으면 계속 다운을 받는다 

#VGG19 
import os, shutil, pathlib 
import tensorflow as tf 
import keras 
import pickle 
from keras import models, layers 
import numpy as np 
import matplotlib.pyplot as plt 

original_dir = pathlib.Path("./dogs-vs-cats/train")
new_base_dir = pathlib.Path("./dogs-vs-cats/dogs-vs-cats_small")

#폴더로 옮기기 => 데이터셋은 폴더를 지정하면 자동으로 라벨링을 한다.폴더이름을 오름차순으로 정렬해서 자동 라벨링 
#데이터셋 -   train -     cats 
#                        dogs
#            test -      cats 
#                        dogs
#           validation - cats 
#                        dogs

#폴더지정, 시작인덱스, 종료인덱스 
def make_subset(subset_name, start_index, end_index): #make_subset("train", 0, 1000)
    for category in ("cat", "dog"):
        dir = new_base_dir/subset_name/category #WindowsPath 라는 특별한 객체, str 아님  
        os.makedirs(dir, exist_ok=True) #디렉토리가 없을 경우 새로 디렉토리를 만들어라 
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)] 
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname) 

make_subset("train", 0, 1000)
make_subset("validation", 1000, 1500)
make_subset("test", 1500, 2000)

from keras.utils import image_dataset_from_directory 

#batch_size에 지정된 만큼 폴더로부터 이미지를 읽어온다. 크기는 image_size에 지정한 값으로 가져온다 
#훈련셋을 쪼개서 8:2 정도로 검증셋을 따로 만드는 방법도 있고 , subset 속성, seed를 이용해 나눠야한다. 
train_ds = image_dataset_from_directory(
    new_base_dir/"train", 
    image_size=(180,180),
    batch_size=16
)

validation_ds = image_dataset_from_directory(
    new_base_dir/"validation", 
    image_size=(180,180),
    batch_size=16
)

test_ds = image_dataset_from_directory(
    new_base_dir/"test", 
    image_size=(180,180),
    batch_size=16
)

#vgg19 이미지 모델 가져오기 
from keras.applications.vgg19 import VGG19 

def deeplearning():
    conv_base = keras.applications.vgg19.VGG19(
        weights="imagenet", 
        include_top=False, #CNN만 가져와라 , CNN이 하단에 있음  , 상단-완전연결망(분류) 
        input_shape=(180, 180, 3) #입력할 데이터 크기를 주어야 한다   
        #데이터셋에서 지정한 크기와 일치해야 한다 , 3-색정보 
    ) 

    conv_base.summary() #CNN요약확인하기 
    #block5_pool (MaxPooling2D)   (None, 5, 5, 512)    
    
    conv_base.trainable = True 
    print("합성공 기반 층을 동결하기 전의 훈련 가능한 가중치 개수 ", len(conv_base.trainable_weights))
    conv_base.trainable = False #동결 
    print("합성공 기반 층을 동결 후의 훈련 가능한 가중치 개수 ", len(conv_base.trainable_weights))

#미세조정 : 
    conv_base.trainable = True 
    for layer in conv_base.layers[:-4]: #상위 4계층만 다시 훈련하게 한다
        layer.trainable = False  #하위계층동결 
    #학습률을 아주 낮춰서 학습을 해야 한다. 
    #optimizer 에 단순 rmsprop를 하면 안된다.  

    #데이터 증강에 필요한 파라미터들 
    data_argumetation = keras.Sequential( [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2), 
        layers.RandomZoom(0.4)
    ])

    #모델 만들기 
    inputs = keras.Input(shape=(180, 180, 3)) #모델의 입력레이어 정의 
    x = data_argumetation( inputs )   #입력이미지에 데이터 증강을 적용한다
    x = keras.applications.vgg19.preprocess_input(x)  #vgg19에 맞는 전처리작업(픽셀값범위조정등)

    #인라인방식으로 cnn 연결하기 
    x = conv_base(x) #특성추출이 이뤄진다. 오래걸린다 
    ############################################ 
    x = layers.Flatten()(x) 
    x = layers.Dense(256)(x) 
    x = layers.Dense(128)(x)
    x = layers.Dense(64)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs) 
    model.compile( loss='binary_crossentropy', 
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
                  metrics = ['accuracy'])
    
    #시스템이 내부적으로 일 처리하고 일 끝나면 우리가 전달해준 콜백 함수를 호출한다. 
    callbacks = [
        keras.callbacks.ModelCheckpoint( 
            filepath="개고양이사전학습2.keras",
            save_best_only=True,
            monitor='val_loss' #검증데이터 셋을 기준으로 하겠다 가장 적절한시점에 호출할거다 
        )
    ]

    history = model.fit(train_ds
                        , epochs=100
                        , validation_data=validation_ds
                        ,callbacks=callbacks)
    
    with open("개와고양이사전학습2.bin", "wb") as file:
        pickle.dump(history.history, file)



deeplearning()