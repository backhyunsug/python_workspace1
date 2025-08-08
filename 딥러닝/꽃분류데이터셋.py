import os, shutil
import pickle 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
# 이미지 전처리 유틸리티 모듈
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

import numpy as np 
import os 
import random 
import PIL.Image as pilimg 
import imghdr
import pandas as pd 
import tensorflow as tf 
import keras 

def study():
    batch_size=32 #한번에 불러올 이미지 개수 
    img_height = 180 #내가 지정한 크기로 이미지를 가져온다  
    img_width  = 180 

    #데이터 증강을 위한 파라미터 지정하기 
    data_augmentation = keras.Sequential(
            [
                    layers.RandomFlip("horizontal"),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
            ]
    )

    model = models.Sequential() 
    model.add(data_augmentation) 
    #증강에 대한 파라미터를 주면 1에포크마다 데이터를 조금씩 변형을 가해서 가져간다
    #과대적합을 막기 위해서 
    model.add(layers.Rescaling(1./255))  #1/255 파이썬 3에서는 결과가 실수 2에서는 정수
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten()) #CNN과 완전연결망을 연결한다.
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',  #라벨원한코딩안하려고 
                  metrics=['accuracy'])
    
    train_dir = "./data/flowers"
    train_ds = keras.preprocessing.image_dataset_from_directory( 
        train_dir, 
        validation_split=0.2,
        subset='training', 
        seed=1234,  
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = keras.preprocessing.image_dataset_from_directory( 
        train_dir, 
        validation_split=0.2,
        subset='validation', #전체 데이터를 20% 나눠서 검증셋으로  
        seed=1234,  
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    history = model.fit(train_ds, 
              epochs=30,
              validation_data = val_ds)
    #모델을 저장하자 
    model.save("flowers_model.keras")
    f = open("flowers_hist.hist", "wb") 
    pickle.dump(history.history, file=f) #주의사항 : history만 저장하면 에러발생 
    f.close()

def drawChart():
    #히스토리를 읽어서 차트 그리기 
    f = open("flowers_hist.hist", "rb")
    history = pickle.load(f)
    f.close()
    print(history.keys()) #키값을 보여줌 

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    #'bo -- blue 동그리미  
    plt.plot(epochs, acc, 'bo', label='Training acc' )
    plt.plot(epochs, val_acc, 'b', label='Validation acc' )
    plt.title('Traing and validation accuray') 
    plt.legend()

    plt.figure() #창하나 더 띄우고->손실값을 띄우겠다 
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show() 

def main():
    while(True):
        print("1.기본학습")
        print("2.차트")
        print("3.예측")
        print("4.평가하기")
        sel = input("선택 : ")
        if sel=="1":
            study() 
        elif sel=="2":
            drawChart()
        elif sel=="3":
            predict()
        else:
            return 

def predict():
    #1. 모델 가져오기 
    model = load_model("flowers_model.keras")
    test_dir="./data/flowers/test" 
    test_ds = keras.preprocessing.image_dataset_from_directory( 
        test_dir, 
        seed=1234,  
        image_size=(180, 180),
        batch_size=1  #한개씩 가져오기, 배치사이즈 단위로 업어와서 만약에 여기에 1이 아닌값을 넣으면    
    )                 #한번에 여러개씩 가져온다. 예측할 데이터 셋이 만을때는 배치에 1이상의 값을 주고 
                      #내부 for문 다시 처리하기  
    print("예측하기")
    print("데이터 개수 ",  tf.data.Dataset.cardinality(test_ds).numpy())  

    i=0
    match_cnt = 0 
    for images, labels in test_ds:
        #테스트데이터셋으로부터 이미지들과 라벨즈를 갖고 온다 
        output=model.predict(images)
        print("라벨 :", labels, np.argmax(output, axis=1), len(output))
        if labels[i] == np.argmax(output): #softmax 라벨이 확률로 온다, 그중에 가장 큰값의 인덱스
            match_cnt+=1 
    print("일치숫자", match_cnt)
    print("불일치개수",  tf.data.Dataset.cardinality(test_ds).numpy()- match_cnt)

if __name__=="__main__":
    main()







