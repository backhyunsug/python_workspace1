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
              epochs=10,
              validation_data = val_ds)
    #모델을 저장하자 
    model.save("flowers_model.keras")
    f = open("flowers_hist.hist", "wb") 
    pickle.dump(history.history, file=f) #주의사항 : history만 저장하면 에러발생 
    f.close()

study()




