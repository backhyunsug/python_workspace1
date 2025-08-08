from keras.datasets import reuters
from keras import models
from keras import layers 
import os

#즉시실행 모드 끄고 하자, 메모리 부족 문제 발생시   
#tf.compat.v1.disable_eager_execution()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(train_data, train_labels), (test_data, test_labels)= reuters.load_data(num_words=10000)
print( train_data.shape)
print( train_labels.shape)
print( test_data.shape)
print( test_labels.shape)

print( train_data[0])

#원래의 단어로 바꾼다 
# word_index는 단어와 정수 인덱스를 매핑한 딕셔너리입니다
word_index = reuters.get_word_index()
# 정수 인덱스와 단어를 매핑하도록 뒤집습니다
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 리뷰를 디코딩합니다. 
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다.케라스 만든사람들이 필요에 의해서 만들음
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


print(decoded_review )



import numpy as np

#원핫인코딩으로 데이터를 변환한다 
def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension))이고 모든 원소가 0인 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다
    return results


# 훈련 데이터를 벡터로 변환합니다
x_train = vectorize_sequences(train_data)
# 테스트 데이터를 벡터로 변환합니다
x_test = vectorize_sequences(test_data)

"""
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# 훈련 레이블 벡터 변환
one_hot_train_labels = to_one_hot(train_labels)
# 테스트 레이블 벡터 변환
one_hot_test_labels = to_one_hot(test_labels)

"""
from keras.utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


print( one_hot_train_labels)

#신경망 만들기 
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,))) #입력층
model.add(layers.Dense(64, activation='relu'))#히든레이어
model.add(layers.Dense(46, activation='softmax')) #출력층

#모델 컴파일 
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#훈련검정셋을 만든다 
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


#학습을 시작한다 
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=500,
                    batch_size=512,
                    validation_data=(x_val, y_val))


predictions = model.predict(x_test)

# 7. 예측 결과 확인 (0번 뉴스 기사)
predicted_class = np.argmax(predictions[0])
actual_class = test_labels[0]

print(f"예측 결과: {predicted_class}, 실제 정답: {actual_class}")


for i in range(20):
    pred = np.argmax(predictions[i])
    actual = test_labels[i]
    print(f"[{i}] 예측: {pred} / 실제: {actual}")

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 그래프를 초기화합니다

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


results = model.evaluate(x_val, y_val)
print(results)

#학습중의 history를 갖고 있다
history_dict = history.history
print( history_dict.keys())




