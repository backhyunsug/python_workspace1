import numpy as np 
import os 
import random 
import PIL.Image as pilimg 
import imghdr
import pandas as pd 
import tensorflow as tf 

#데이터만들기 folder 를 읽어서 데이터를 만들어보자 - train 폴더 
base_path = "./data/flowers2"
def makeData(flower_name, label, isTrain=True):  #train/daisy 0     train/dandelion 1 
    if isTrain:
        path = base_path+"/train/"+ flower_name 
    else:
        path = base_path+"/test/"+ flower_name 
    #print(path)
    data = []
    labels = [] 
    #print( os.listdir(path)) #해당 경로에 파일명을 모두 가져온다 
    #파일 하나씩 읽어서 넘파이배열로 만들어서 data 에 추가시키기
    i=1 
    for filename in os.listdir(path):
        try:
            if i%100==0:
                print(f"{i} 번째파일 처리중 .... ")
            i+=1

            #파일 속성도 확인해보자 
            kind = imghdr.what(path+"/"+filename)
            if kind in ["gif", "png", "jpeg", "jpg"]: #이미지일때만 
                img = pilimg.open(path+"/"+filename)  #파일을 읽어서 numpy 배열로 바꾼다 
                resize_img = img.resize((80,80)) #사이즈는 특성이 너무 많으면 계산시간도 오래걸리고 
                                                 #크기가 각각이면 학습불가능 그래서 크기를 맞춘다 
                pixel = np.array(resize_img)
                if pixel.shape == (80,80,3):
                    data.append(pixel)
                    labels.append(label)    
        except:
            print(filename + " error")       

    title = "train"
    if not isTrain:
        title = "test"
    savefileName = "imagedata{}.npz".format(str(label)+'_'+title)
    np.savez(savefileName, data=data, targets=labels)

def initData():
    flowers = ["daisy", "dandelion"]
    i=0
    for f in flowers:
        makeData(f, i, True) 
        makeData(f, i, False) 
        i+=1 

def loadData():
    f1 = np.load("imagedata0_train.npz")
    f2 = np.load("imagedata1_train.npz")
    d1 = f1["data"]
    t1 = f1["targets"]
    d2 = f2["data"]
    t2 = f2["targets"]
    #print(d1[:10])
    print(t1[:10])
    
    X_train = np.concatenate( (d1, d2), axis=0)
    y_train = np.concatenate( (t1, t2), axis=0)

    f1 = np.load("imagedata0_test.npz")
    f2 = np.load("imagedata1_test.npz")
    d1 = f1["data"]
    t1 = f1["targets"]
    d2 = f2["data"]
    t2 = f2["targets"]

    X_test = np.concatenate( (d1, d2), axis=0)
    y_test = np.concatenate( (t1, t2), axis=0)

    return X_train, y_train, X_test, y_test 
    
from tensorflow.keras import models, layers 

def createModel():
    network = models.Sequential(
        [
            layers.Conv2D(64, (3,3), activation='relu'),  #층은 내마음대로 한거임 
            layers.MaxPooling2D((2,2)),                   #결과가 과대적합이 되었으면 조절을 해야 한다. 
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.Flatten(), 
            layers.Dense(128, activation="relu"),
            layers.Dense(32,  activation="relu"),
            layers.Dense(1,   activation="sigmoid") 
            #2진분류일때는 결과값 1개이고, 활성화함수는 sigmoid 
            #암인지 아닌지   target이 악성:0, 양성:1 
            #1이 될 확률을 출력 => 예측확률 => 양성일 확률이 출력된다. 
            #악성이 될 확률은 1 - 양성확률 
            #target을 0->1 로 바꾸고 1->0으로 바꾸거나 
            #2, softmax 
            #loss 이진분류일때는 binary_crossentropy 
        ]
    )

    network.compile( optimizer='rmsprop', 
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    return network 

from sklearn.preprocessing import StandardScaler

def preprocessing():
    X_train, y_train, X_test, y_test = loadData()
    #차원이   현재 4차원 => 2차원으로 
    # X_train = X_train.reshape(X_train.shape[0],  80*80*3 )
    # X_test  = X_test.reshape(X_test.shape[0],  80*80*3 )

    X_train_scaled = X_train/255
    X_test_scaled = X_test/255

    return X_train_scaled, y_train, X_test_scaled, y_test 

def main():
    X_train, y_train, X_test, y_test = preprocessing() 
    network = createModel() 
    network.fit(X_train, y_train, epochs=100, batch_size=100)

    train_loss, train_acc  = network.evaluate(X_train, y_train)
    test_loss, test_acc  = network.evaluate(X_test, y_test)
    print(f"훈련셋   손실값 :{train_loss}  정확도:{train_acc}")
    print(f"테스트셋 손실값 :{test_loss}  정확도:{test_acc}")

if __name__ == "__main__":
    main()
    
    #initData()
    #X_train, y_train, X_test, y_test = loadData()
    # print(X_train.shape)
    # print(X_train[:5])
    # print(y_train.shape)
    # print(y_train[:5])
    # print(X_test.shape)
    # print(X_test[:5])
    # print(y_test.shape)
    # print(y_test[:5])

#한시간 => 젤 높은 점수한테 커피쿠폰, 훈련셋하고 테스트셋 차이도 작고       
        

