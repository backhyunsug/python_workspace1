from tensorflow.keras.datasets  import mnist #우편번호 손글씨 
import tensorflow as tf
from tensorflow import keras 

tf.random.set_seed(1234)

#1.데이터 가져오기 
(train_images, train_labels),(test_images, test_labels)= mnist.load_data() 
print(type(train_images), type(train_labels))
print(train_images.shape, train_labels.shape) #(60000, 28, 28) (60000,)
print(test_images.shape,  test_labels.shape)  #(10000, 28, 28) (10000,)  
#처음시작할때 70000개 다운받음 
#conda activate deeplearning 

#2.딥러닝 모델을 만든다. 
from tensorflow.keras import models, layers 

#네트워크 또는 모델이라고 부른다 
#keras.Sequenctial 로 모델을 만드는데 매개변수로 list타입안에 레이어 객체를 전달한다 

model = keras.Sequenctial([
    #2-1. 입력층을 설계한다 
    #layers.Dense(출력값의개수, 활성화함수, 입력데이터의 크기-생략가능)
    #출력값의 개수? 저 계층을 나왔을때가 가져올 가중치들의 개수 내마음대로  너무  크게 주면
    #메모리 부족도 있고 , 과대적함 문제도 있음, 적당히, 2의 배수로 많이들 준다   
    layers.Dense(64, activation='relu'),
    #2-2 중간에 다른층 추가 가능 
    #2-3 출력층, 마지막 층은 라벨에 맞춘다. 즉 결과를 얻기 위한 층이다. 
    #    손으로 쓴 숫자니까 0, ~9 까지 10개중에 하나이어야 한다. 딥러닝의 분류는 출력데이터를 
    #    확률로 반환한다. 예) [0.1, 0.1, 0.05, 0.7, .......]  결과는 3으로 판단한다
    #    각 층을 거치면서 나오는 값들은 실제는 확률이 아니고 엄청큰 값들이다. 이걸 
    #    모두합해서 1이 되는 확률로 전환해야하는데 이 함수가 softmax함수이다.
    #    다중분류의 출력층의 활성화 함수는 무조건 softmax함수이다  
    layers.Dense(10, activation='softmax') #출력값의개수, 활성화함수
    #회귀랑 이진분류랑 다중분류랑 다 다르게 자성해야 한다. 회귀는 출력결과 1개만 
    #회귀의 경우 출력층 :  layers.Dense(1),
    #이진분류의 경우 출력층 : layers.Dense(1,activation='sigmoid')
])
#네트워크 또는 모델을 만들다라고 표현한다 
#compile 에서 , 손실함수, 옵티마이저, 평가지표를 지정한다. 
#sparse_categorical_crossentropy 희한하게 딥러닝은 라벨도 원핫인코딩 
#머신러닝은 라벨을 원핫인코딩 , 원핫인코딩 자동으로 해준다 
model.compile( optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy',
               metrix=['accruacy'])




