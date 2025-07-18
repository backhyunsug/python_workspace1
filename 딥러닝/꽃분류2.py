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
    for filename in os.listdir(path):
        try:
            print(filename + "....") 
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
    np.savez(savefileName, data=data, targets=label)

if __name__ == "__main__":
    makeData("daisy", 0, True) 
    # makeData("daisy", 0, False) 
    # makeData("dandelion", 0, True) 
    # makeData("dandelion", 0, True) 
    