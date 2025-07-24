#꽃숫자 세기 

import os 
base_path ="./data/flowers"
print( "daisy", len( os.listdir(base_path+"/daisy")) ) 
print( "dandelion", len( os.listdir(base_path+"/dandelion")) )
print( "sunflower", len( os.listdir(base_path+"/sunflower")) )
print( "rose", len( os.listdir(base_path+"/rose")) )
print( "tulip", len( os.listdir(base_path+"/tulip")) )      


#VGG19 
import os, shutil, pathlib 
import tensorflow as tf 
import keras 
import pickle 
from keras import models, layers 
import numpy as np 
import matplotlib.pyplot as plt 

original_dir = pathlib.Path("./data/flowers")
new_base_dir = pathlib.Path("./data/new_flowers")

#폴더로 옮기기 => 데이터셋은 폴더를 지정하면 자동으로 라벨링을 한다.폴더이름을 오름차순으로 정렬해서 자동 라벨링 


#폴더지정, 시작인덱스, 종료인덱스 
def make_subset(subset_name, startIndex=0, endIndex=700): #make_subset("train", 0, 1000)
    for category in ("daisy", "dandelion", "tulip", "rose", "sunflower"):
        dir = new_base_dir/subset_name/category #WindowsPath 라는 특별한 객체, str 아님  
        os.makedirs(dir, exist_ok=True) #디렉토리가 없을 경우 새로 디렉토리를 만들어라 
        #파일명ㅇ cats.o.jpg cats.1.jpg ........
        dataList = os.listdir(base_path+"/"+category) #리스트를 가져와서 
        if endIndex != -1: 
            fnames = dataList[startIndex:endIndex] 
        else: #데이터개수가 몇개인지 몰라서 endIndex값이 -1이 오면 
            fnames = dataList[startIndex:]
        for fname in fnames:
            shutil.copyfile(src=original_dir/category/fname, dst=dir/fname) 
#데이터가 적어서 700건 까지만 훈련셋으로 함 
make_subset("train", 0, 700)
make_subset("test", 700, -1) #700건 넘는 데이터를 테스트셋으로 보냈음 

from keras.utils import image_dataset_from_directory 

#batch_size에 지정된 만큼 폴더로부터 이미지를 읽어온다. 크기는 image_size에 지정한 값으로 가져온다 
#훈련셋을 쪼개서 8:2 정도로 검증셋을 따로 만드는 방법도 있고 , subset 속성, seed를 이용해 나눠야한다. 
print(new_base_dir) #트레인에 있는 데이터를 훈련셋과 검증셋으로 쪼갰음 
#데이터가 많이 않아서 검증 폴더를 따로 쪼갤 수 없을때
train_ds = image_dataset_from_directory(
    new_base_dir/"train",
    seed=1234,           #seed :동일하게 자르게 하기위해서
    subset='training',   #훈련셋
    validation_split=0.2, #얼마만큼의 비율로 자를거냐 
    image_size=(180,180),
    batch_size=16
)

validation_ds = image_dataset_from_directory(
    new_base_dir/"train",
    seed=1234,
    subset='validation',
    validation_split=0.2, 
    image_size=(180,180),
    batch_size=16
)

# validation_ds = image_dataset_from_directory(
#     new_base_dir/"validation", 
#     image_size=(180,180),
#     batch_size=16
# )

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
    outputs = layers.Dense(5, activation='softmax')(x)  ###############

    model = keras.Model(inputs, outputs) 
    model.compile( loss='sparse_categorical_crossentropy', ##############
                  optimizer='rmsprop', 
                  metrics = ['accuracy'])
    
    #시스템이 내부적으로 일 처리하고 일 끝나면 우리가 전달해준 콜백 함수를 호출한다. 
    callbacks = [
        keras.callbacks.ModelCheckpoint( 
            filepath="flowers2.keras",
            save_best_only=True,
            monitor='val_loss' #검증데이터 셋을 기준으로 하겠다 가장 적절한시점에 호출할거다 
        )
    ]

    history = model.fit(train_ds
                        , epochs=5
                        , validation_data=validation_ds
                        ,callbacks=callbacks)
    
    with open("flowers2.bin", "wb") as file:
        pickle.dump(history.history, file)



deeplearning()
