#딥러닝 , 회귀, 이진분류, 다중분류 다 된다. 
import tensorflow
from tensorflow.keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data() 

print(X_train.shape) 
print(X_train[:5])
print(y_train.shape) 
print(y_train[:5])

#스케일링 작업만 
from sklearn.preprocessing import Normalizer 
normal = Normalizer() 
X_train_scaled = normal.fit_transform(X_train)
X_test_scaled = normal.fit_transform(X_test)

from tensorflow.keras import models, layers 
def makeModel():
    model = models.Sequential(
        [
            layers.Dense(512, activation='relu', input_shape=((13,))), #input_shape=((13,) 생략가능하다 
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64,  activation='relu'),
            
            #맨마지막에 회귀의 경우에 확률을 가져올게 아니고 연산결과 하나만 가져온다. 
            layers.Dense(1)
        ]
    )

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'] )

    return model 

network = makeModel() 
network.fit(X_train_scaled, y_train, epochs=10, batch_size=100)
train_loss, train_acc  = network.evaluate(X_train_scaled, y_train)
test_loss, test_acc  = network.evaluate(X_test_scaled, y_test)
print(f"훈련셋   손실값 :{train_loss}  mae:{train_acc}")
print(f"테스트셋 손실값 :{test_loss}  mae:{test_acc}")


