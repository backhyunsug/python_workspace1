#1.파일을 읽어서 numpy 배열로 만들어야 한다 
#2.폴더별로 labeling 을 해야 한다 

import numpy as np 
import os 
import PIL.Image as pilimg #이미지 파일을 읽어서 numpy 배열로 바꾼다
import imghdr #이미지의 종류를 알아내고자 할때 사용
import pandas as pd

#makeData =>폴더명하고 라벨을 주면    makeData("daisy", 1)
#해당 폴더 데이터 쭉 읽어서 numpy 배열로 바꾸고 라벨링 작업도 하고 
def makeData(folder, label):
    data =[]  #이미지의 피처를 저장  - 마지막에 둘다 반환하기 
    labels=[] #라벨 저장
    path="./data/flowers"+"/"+folder 
    for filename in os.listdir(path):
        try:
            print(path+"/"+filename)
            kind = imghdr.what(path + "/"+ filename)#파일의 종류를 확인하기 위한 파이썬 명령어 
            #파일의 확장자를 잘라서 확인해도 되는데? 확장자는 윈도우 os만 있지 리눅스 파일정보에 종류저장
            if kind in ["gif", "png", "jpg", "jpeg"]: #이 이미지에 해당되면
                img = pilimg.open(path + "/"+ filename)
                #이미지 크기가 다르면 분석 불가능, 동일한 이미지 크기로 해야 한다 
                #그리고 이미지 크기가 너무 크면 학습할때 픽처 개수가 너무 많아서 학습이 어렵다 
                #적당한 크기로 잘라야 한다 
                resize_img = img.resize( (80, 80) ) #크기를 튜플로 전달하면 앞의 이미지크기만 변경됨 
                pixel = np.array(resize_img)  #이미지를 => narray로 변경한다 
                if pixel.shape==(80,80,3):  #이미지 크기 같은것만 취급한다 
                    data.append(pixel)
                    labels.append(label)
        except:
            print(filename + " error ") #어떤파일이 에러인지 찾아서 직접 삭제시킬려고 

    #파일로 저장하기 
    np.savez("{}.npz".format( folder), data=data, targets=labels)


def filesave():
#1.파일로 저장하기 
    makeData("daisy", "0") 
    makeData("dandelion", "1") 
    makeData("sunflower", "2") 
    makeData("rose", "3") 
    makeData("tulip", "4") 

def loadData():
    daisy=np.load("daisy.npz")
    dandelion=np.load("dandelion.npz")
    sunflower=np.load("sunflower.npz")
    rose=np.load("rose.npz")
    tulip=np.load("tulip.npz")

    data = np.concatenate((daisy["data"], dandelion["data"], sunflower["data"],rose["data"], \
                           tulip["data"] ))
    target=np.concatenate((daisy["targets"], dandelion["targets"], sunflower["targets"], \
                           rose["targets"],  tulip["targets"] ))
    print(data.shape)
    print(target.shape)
    return data, target #지역변수라서 함수안에만 존재 리턴을 해주자
 
data, target = loadData()    
data = data.reshape(data.shape[0], 80*80*3 ) #4차원 머신러닝, 딥러닝 => 딥러닝(CNN,합성곱신경망)
#차원을 그대로 받아들인다.  
print(data.shape)
#data = 4317, 80, 80, 3

#머신러닝의 일부 알고리즘과 딥러닝은 반드시 정규화 또 스케일링 0~1 사이의 값으로 축소시키는것이 유리함
data = data/255.0

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0, test_size=0.5)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("훈련셋", model.score(X_train, y_train))
print("테스트셋", model.score(X_test, y_test))

