from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras import layers, models   
tf.random.set_seed(1)

iris = load_iris()
X = iris['data']
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

#데이터 처리 -> 차원변환, 스케일링
print(X_train.shape) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.fit_transform(X_test)

#딥러닝은 라벨도 원핫인코딩 해야 한다 
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

#네트워크 구축하기 
network = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu'), 
    keras.layers.Dense(64, activation='relu'), #히든충 추가 - 내마음
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(3, activation='softmax') 
])

#컴파일 옵티마이저, 손실함수, 평가척도 지정 
network.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#학습하기
history = network.fit(X_train_scaled, y_train, epochs=30, batch_size=100)

#평가하기
train_loss, train_acc = network.evaluate(X_train_scaled, y_train) 
print(f"훈련셋 손실:{train_loss}  정확도{train_acc}")
test_loss, test_acc = network.evaluate(X_test_scaled, y_test) 
print(f"테스트셋 손실:{test_loss}  정확도{test_acc}")
