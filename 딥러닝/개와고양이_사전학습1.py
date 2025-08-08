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

import gdown #케라스만든사람들이 케라스에 있는 데이터셋 업어오기위해 사용함 
#gdown.download(id='18uC7WTuEXKJDDxbj-Jq6EjzpFrgE7IAd', output='dogs-vs-cats.zip')
#안막으면 계속 다운을 받는다 

#VGG19 
import os, shutil, pathlib 
import tensorflow as tf 
import keras 
import pickle 

original_dir = pathlib.Path("./dogs-vs-cats/train")
new_base_dir = pathlib.Path("./dogs-vs-cats/dogs-vs-cats_small")

#폴더로 옮기기 
def make_subset(subset_name, start_index, end_index): #make_subset("train", 0, 1000)
    for category in ("cat", "dog"):
        dir = new_base_dir/subset_name/category
        os.makedirs(dir, exist_ok=True) #디렉토리가 없을 경우 새로 디렉토리를 만들어라 
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)] 
        for fname in fnames:
            shutil.copyfile(src=original_dir/fname, dst=dir/fname) 

make_subset("train", 0, 1000)
make_subset("validation", 1000, 1500)
make_subset("test", 1500, 2000)

from keras.utils import image_dataset_from_directory 

#batch_size에 지정된 만큼 폴더로부터 이미지를 읽어온다. 크기는 image_size에 지정한 값으로 가져온다 
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

conv_base = keras.applications.vgg19.VGG19(
    weights="imagenet", 
    include_top=False, #CNN만 가져와라 , CNN이 하단에 있음  
    input_shape=(180, 180, 3) #입력할 데이터 크기를 주어야 한다   
    #데이터셋에서 지정한 크기와 일치해야 한다 
) 

conv_base.summary() #CNN요약확인하기 
#block5_pool (MaxPooling2D)   (None, 5, 5, 512)    

#데이터셋을 주로 CNN으로부터 특징을 추출해서 전달하는 함수 
import matplotlib.pyplot as plt
import numpy as np 
def get_features_and_lables( dataset): 
    all_feautres=[]
    all_labels=[] 
    for images, labels in dataset: #예측할때처럼 폴더로부터 16개의 이미지와 라벨을가져온다 
        preprocessed_images = keras.applications.vgg19.preprocess_input(images) 
        print(images.shape, preprocessed_images.shape)
        #plt.imshow(preprocessed_images[0])
        #plt.show()
        #break
        features = conv_base.predict(preprocessed_images)
        all_feautres.append(features) 
        all_labels.append(labels)
    return np.concatenate(all_feautres), np.concatenate(all_labels)

def save_features():
    train_features, train_labels = get_features_and_lables(train_ds)     
    validation_features, validation_labels = get_features_and_lables(validation_ds)     
    test_features, test_labels = get_features_and_lables(test_ds)     

    data = [train_features, train_labels, validation_features, validation_labels,
               test_features, test_labels]
    with open('개고양이특성.bin', 'wb') as file:
        pickle.dump(data, file)
    
def load_features():
    with open('개고양이특성.bin', 'rb') as file:
        data = pickle.load(file)

    return data[0], data[1], data[2], data[3], data[4], data[5] 
    
    

from keras import models, layers 
def deeplearning():
    train_features, train_labels, validation_features, validation_labels , \
    test_features, test_labels= load_features()

    #특성추출, 불러오기, 예측 
    data_argumentation = keras.Sequential(
            [
		            #layers.RandomFlip("horizontal", input_shape=(180, 180, 3)),
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
            ]
    )
    
    #맨 마지막 block
    inputs = keras.Input(shape =(5,5,512))
    x = data_argumentation(inputs)
    x = layers.Flatten()(x) 
    x = layers.Dense(256)(x) 
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    #학습이 과소가 되었던 과대가 되었던 학습이 끝나야 저장이 되는데 
    #과대적합이 되는 시점에서 저장을 할 수 있다 
    #model이 학습하는 도중에 과대적합이 되는걸 확인할 수 있다. 
    #콜백함수에 저장할 파일명을 전달하면 자동으로 호출을 한다. 
    #list 형태로 받아간다
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="특성추출.keras", 
            save_best_only=True, #가장 적합할때 저장하기 
            monitor="val_loss" #검증데이터의 손실로스값이 최적화일때  
        )
    ]

    model.compile(loss='binary_crossentropy', 
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = model.fit(train_features, train_labels, 
                        epochs=5, 
                        validation_data=(validation_features, validation_labels),
                        callbacks=callbacks)     

def predict():
    model = keras.models.load_model("특성추출.keras")
    train_features, train_labels, validation_features, validation_labels , \
    test_features, test_labels= load_features()

    test_pred = model.predict(test_features)
    test_pred = (test_pred>0.5).astype("int").flatten() 
    print(test_pred[:20])
    print(test_labels[:20])
    match_count = np.sum(test_pred == test_labels)
    print("맞춘개수 : ",  match_count)
    print("틀린개수 : ", len(test_labels)-match_count )

    # for i in range(0, 10):
    #     print(test_labels[i], test_pred[i], test_labels[i]==test_pred[i])
    
def main():
    while True:
        print("1.특징추출")
        print("2.학습")
        print("3.예측")
        sel = input("선택 ")
        if sel=="1":
            save_features() 
        elif sel=="2":
            deeplearning()
        elif sel=="3":
            predict() 
        else:
            break 

#예측함수 => test_features, test_labels  

if __name__ == "__main__":
    main()