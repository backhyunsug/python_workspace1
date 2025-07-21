#fationminst 자료 
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from tensorflow.keras import models, layers 
img_height=28
img_width =28
network = models.Sequential(
    [
        #스케일링은 반드시, 내부 스케일링 사용
        #차원은 그대로   
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)), 
        layers.Conv2D(32, (3,3), activation='relu'),
        #32 - 출력값 , (3,3) - 필터의 크기 보통 이값을 사용한다 
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling((2,2)), #서브샘플링, 특성의 개수를 줄여서 과대적합을 방지한다
        
        #CNN와 완전연결망을 연결하기 위한 계층 4차원 => 2차원으로 바뀌어야 한다 
        layers.Flattern(),

        #완전연결망 
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') #마지막 출력층 
    ]
)

import tensorflow as tf 
network.compile(optimizer='adam', 
                loss=tf.keras.SparseCategoricalCrossentrophy(),  #라벨이 정수형일때)
                metrics = ['accuracy'])

print(network.summary())

network.fit( X_train, y_train, epochs=10, validation_split=0.2)

print("훈련셋", network.evaluate(X_train, y_train))
print("테스트셋", network.evaluate(X_test, y_test))


