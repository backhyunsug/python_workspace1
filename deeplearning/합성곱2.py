#칼라그림 가져다가 일반 딥러닝과 CNN비교하기
#칼라그림도 numpy배열로 잘 정리해서 준다 
from keras.datasets import cifar10 
import numpy as np 
import tensorflow as tf 
import keras 
from keras.utils import to_categorical 

(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
#50000개의 훈련셋, 10000개의 테스트셋이 있음 
print(X_train.shape)
print(y_train.shape)
print( len(np.unique(y_train))) #카테고리가 몇개인지 확인 필요

#이미지 출력코드 
import matplotlib.pyplot as plt 

def imageShow(id):
    image = X_train[id]
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()

imageShow(0) 

#이미지 여러개 보기 
def imageShow2(train_images, row, col):
    plt.figure(figsize=(10, 5))
    for i in range(row*col):
        plt.subplot(row,col, i+1)
        image = train_images[i]
        plt.imshow(image, cmap= plt.cm.binary)
    plt.show()

imageShow2(X_train, 5, 5)

from keras import models, layers 
#이번에는 CNN아닌걸로 
def make_model1():
    network = models.Sequential([
        layers.Dense(128, activation='relu'), #세번째인자 input_shape 입력차원지정, 현재설치버전은 삭제
        layers.Dense(128, activation='relu'), 
        layers.Dense(10,  activation='softmax'), 
    ])
 
    network.compile( optimizer='sgd', 
                    loss = 'categorical_crossentropy',
                     metrics=['accuracy']) 

    (X_train, y_train), (X_test, y_test) = cifar10.load_data() 
    #cnn 아닐때 차원변경 
    X_train = X_train.reshape(50000, 32*32*3 )
    X_test  = X_test.reshape(10000, 32*32*3 )
    
    #스케일링 
    X_train = X_train/255 
    X_test  = X_test/255 

    #라벨은 원핫인코딩을 해야 한다 
    y_train = to_categorical(y_train)
    y_test =  to_categorical(y_test)

    print("학습시작하기")
    network.fit(X_train, y_train, epochs=100, batch_size=100 )

    #머신러닝 score 함수 대신에 평가 
    train_loss, train_acc = network.evaluate( X_train, y_train)
    print("훈련셋 손실 {} 정확도 {}".format(train_loss, train_acc))

    test_loss, test_acc = network.evaluate(X_test,   y_test)
    print("테스트셋 손실 {} 정확도 {}".format(test_loss, test_acc))


if __name__ == "__main__":
    make_model1()
