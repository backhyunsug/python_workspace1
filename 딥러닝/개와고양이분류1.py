#폴더에 train, test, validation 등으로 나눈다음에 각자에 폴더에 라벨을 만들고 
#데이터 넣어놓고, ImageDataGenerator 나 DataSet 을 통해서 파일을 직접 읽어서 
#학습한다. 데이터 증강  ImageDataGenerator(초창기부터)-폴더로부터 직접 이미지 파일
#을 읽어서 각종 처리를 해서 원하는 만큼 데이터를 늘려서 가져온다.
#좀더 정밀하게 비슷한 일을 한다. DataSet - Tensor 2.xx이후 추가 
#이미지 => numpy배열로 바꿔서 학습:데이터가 충분히 많으면 

#데이터셋이 작을때 이미 학습된 CNN하고 작업할때 쓸 예정임 
#1.cats_and_dogs_small - train, test, validation 만들어서 
#  train/cat- 고양이사진 1000장 dog-개사진 1000장 
#  test/cat- 고양이사진 500장 dog-개사진 500장
#  validation/cat- 고양이사진 500장 dog-개사진 500장

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import random 
import PIL.Image as pilimg 
import imghdr
import pandas as pd 
import pickle 
import keras 
import os 
import shutil #디렉토리 만들기, 삭제등을 담당하는 라이브러리

#원본데이터셋이 있는 위치 경로 
original_dataset_dir = "./data/cats_and_dogs/train"

#옮길위치 - 기본 폴더 
base_dir = "./data/cats_and_dogs_small"

train_dir = os.path.join(base_dir, 'train')
test_dir   = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

#ImageDataGenerator나 DataSet이나 둘 다 폴더보고 자동으로 라벨링을 한다 
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#학습모델(네트워크) - 학습을 완료한 모델을 저장시켜놓고 불러다가 예측을 할 수 있다 
model_save_path_keras = 'cat_and_dogs_model.keras' #확장자가 h5=>keras로 바뀜
#케라스가 지원한다 
history_filepath = 'cats_and_dogs_history.bin' 
#학습을 할때마다 정확도, 손실값이 있는 있는 저장해서 던져준다 
#이 값 자체는 저장을 지원하지 않는다. 그래서 pickle을 써서 저장하는데 
#희한하게도 history 자체로 저장하면 에러나고 history.history를 저장해야 에러가 안난다



def ImageCopy():
    #디렉토리내의 파일개수 알아내기
    totalCount = len(os.listdir(original_dataset_dir))
    print("전체개수", totalCount)

    #바복적인 실행을 위해서 디렉토리 삭제
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True, onerror=None) 

    #디렉토리 만들고
    os.makedirs(base_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(validation_dir)
    
    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)

    #파일 옮기기
    #옮길 파일명이 cat0.jpg, cat1.jpg ,,, cat1000.jpg 
    fnames = [ f'cat.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'cat.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'cat.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사
    
    fnames = [ f'dog.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'dog.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'dog.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

#ImageCopy() - 복사 끝남 

from keras import models, layers
#DataSet 사용하기 
def deeplearning():
    #데이터 증강 파라미터-> 에포크마다 데이터를 조금씩 변형해서 가져온다. 
    #과대적합을 막기 위해서  
    data_augmentation = keras.Sequential(
            [
		            layers.RandomFlip("horizontal", input_shape=(180, 180, 3)),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                    #layers.RandomFlip("horizontal"): 이미지를 수평으로 무작위로 뒤집습니다. 
                    #layers.RandomRotation(0.1): 이미지를 최대 2pi 라디안의 10 (즉, 36도)까지 무작위로 회전시킵니다.
                    #layers.RandomZoom(0.1): 이미지를 최대 10%까지 무작위로 확대하거나 축소합니다.
                ]
        )
    
    model = models.Sequential() 
    #이미지 스켈일링 
    model.add( layers.Rescaling(1./255))
    model.add(data_augmentation) ##########################################################
    model.add(layers.Conv2D(32, (3,3), activation='relu' ))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation='relu' ))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32, (3,3), activation='relu' ))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) #중간에 데이터를 절반쯤 없애서 과대적합을 방지 
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    #데이터셋 - 폴더로 부터 이미지 파일을 읽어온다 
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,  #폴더를 지정한다 
        validation_split=0.2, #훈련셋을 훈련셋과 검증셋으로 8:2구조로 나눠서검증
        seed=123,
        subset="training",
        image_size=(180,180),
        batch_size=16
    )

    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,  #폴더를 지정한다 
        validation_split=0.2, #훈련셋을 훈련셋과 검증셋으로 8:2구조로 나눠서검증
        seed=123,
        subset="validation",
        image_size=(180,180),
        batch_size=16
    )


    history = model.fit(train_ds, 
              validation_data=val_ds,
              epochs=30)
    
    #모델 저장하기 
    try: 
        model.save(model_save_path_keras)#모델 저장하기 
        print("모델 저장 완료")
    except Exception as e:
        print(f"모델 저장중 오류 발생 {e}")

    #히스토리 저장하기 
    try:
        with open(history_filepath, 'wb') as file: 
            pickle.dump(history.history, file)
        print("히스토리 저장 완료")
    except Exception as e:
        print(f"히스토리 저장중 오류 발생 {e}")

def drawChart():
    print("저장된 모듈 불러오기")
    try:
        loaded_model_keras = keras.models.load_model(model_save_path_keras) 
        print("모델 부르기 성공")
    except Exception as e:
        print(f"모델 로딩중 실패 : {e}")

    print("히스토리 불러오기")
    try:
        with open(history_filepath, "rb") as file:
            history = pickle.load(file)
            print("히스토리 로딩 성공")
    except Exception as e:
        print(f"히스토리 로딩중 실패 : {e}")


    #히스토리의 키값들 가져오기 - 에포크회수만큼 list로 가져온다 
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    #x축 좌표값 만들기 
    X = range(len(acc))  

    plt.plot(X, acc, 'ro', label='Training accuracy')
    plt.plot(X, val_acc, 'b', label='Validation accuracy')
    plt.title("Training and Validation accuracy")

    #새로운 창을 열어 차트를 그린다.
    plt.figure() #새로운창을만든다 
    plt.plot(X, loss, 'ro', label='Training loss')
    plt.plot(X, val_loss, 'b', label='Validation loss')
    plt.title("Training and Validation loss")

    plt.show() #전체 한번에 출력

def Predict(): #예측하기 
    #1.학습된 모듈을 불러온다 
    loaded_model_keras = None
    try:
        loaded_model_keras = keras.models.load_model(model_save_path_keras) 
        print("모델 부르기 성공")
    except Exception as e:
        print(f"모델 로딩중 실패 : {e}")
        return 

    #예측데이터셋 
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,  #폴더를 지정한다 
        validation_split=0.2, #훈련셋을 훈련셋과 검증셋으로 8:2구조로 나눠서검증
        seed=123,
        subset="validation",
        image_size=(180,180),
        batch_size=16
    )

    print("----- 라벨링 ------")
    class_names = val_ds.class_names 
    print(class_names)

    total_match_count =0 #전체 일치한 개수 
    total_samples_proceed=0 #전체처리개수 
    max_samples_to_process = 500 #데이터르 500개 까지만 예측해보자 
    for input_batch, labels_batch in val_ds:
        #val_ds 가 폴더로 부터 이미지 파일을 읽어오는데 batch_size만큼씩 읽어온다 
        total_samples_proceed += len(labels_batch)

        #예측하기 
        prdictions_batch = loaded_model_keras.predict(input_batch, verbose=2)
        print(prdictions_batch)

        #예측결과와 실제 레이블 비교 
        #이진분류라서 결과값이 하나가 온다. 꽃분류같으면 맞는다 [0.3, 0.7]  [0.3]
        #이진분류일때는 라벨이 1인 요소의 확률을 전달한다 
        #다중분류일때는 [0.1, 0.1, 0.6, 0.1, 0.1]
        predicted_classes = (prdictions_batch>0.5).astype(int) #0.5보다 큰거는 True, 작은거는 False 
        print("예측 : ", predicted_classes.flatten())
        print("라벨 : ", labels_batch.numpy()) #Tensor -> numpy로
        
        match_count = np.sum(predicted_classes.flatten()== labels_batch.numpy() )
        total_match_count+= match_count

    #print(total_samples_proceed, len(labels_batch))
    print("전체데이터개수 ", total_samples_proceed)
    print("맞춘개수 ", total_match_count)
    print("못맞춘개수 ", total_samples_proceed-total_match_count)
    
def main():
    while True:
        print("1. 파일복사")
        print("2. 학습")
        print("3. 차트")
        print("4. 예측")
        sel = input("선택 : ")
        if sel=="1":
            ImageCopy()
        elif sel=="2":
            deeplearning()
        elif sel=="3":
            drawChart() 
        elif sel=="4":
            Predict() 
        else:
            break 
        

if __name__ == "__main__":
    main()