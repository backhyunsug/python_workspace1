#imdb1.py 
import keras 
from keras.datasets import imdb 
from keras import models 
from keras import layers
import tensorflow as tf 

import tensorflow as tf 
import os

#케라스 입장에서 문자열 데이터들을 어떤 형태로 numpy배열로 만들었는지를 보고 
#imdb 데이터셋 => numpy 배열로 바꿔서 온거 
#문자열들을 어떤식으로  numpy 배열로 바꿀 것인가? (다음주에) 
#영화평들을 다 읽어서 => numpy배열로 바꾼다.(케라스)
#빈도수로 파악할때 자주 쓰는 단어 10000 개만 가져다 쓰겠다 
#num_words=10000 :빈도수를 기반으로 해서 자주 쓰는 단어 만개만 가져다 쓰겠다 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
#데이터 개수 확인
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_data.shape)
#데이터 자체도 궁금 
print(train_data[:3]) #문장을 list타입으로  가져온다
print(train_labels[:3]) 

#데이터를 시퀀스로 바꿔야 하는데, 시퀀스로 바꾸는 과정은 담주 
#get_word_index() 
word_index = imdb.get_word_index() 
print(type(word_index))
#print(word_index.keys())

def showDictionay(cnt): #word_index의 내부구조 확인 
    i=0 
    for key in word_index:
        if i >= cnt: 
            break 
        print(key, word_index[key])
        i += 1 

showDictionay(10) #word_index  영어단어:인덱스 
#I like star.      {"I":0, "like":1,  "star":2 ,............
#[0, 1, 2]

#받아온 시퀀스를 문장으로 원복 시켜  보자  word_index는 단어:숫자 
#reverse_index   => 숫자:단어 
reverse_index = [(value, key) for(key, value) in word_index.items()]
for i in range(0, 10):
    print(reverse_index[i])
#dict으로 바꿔야 한번에 검색함 
reverse_index = dict(reverse_index)

#시퀀스를 문장으로 바꾸기 
#케라스만든 사람들이 0~3 번까지는 특수목적으로  인덱스 4부터 쓸모가 있음 
def changeSentence(id):
    sequence = train_data[id]
    #10000 개만 가져오니까 없는 단어도 있을 수 있음 그때 두번째인자로 바꿔준다. 
    sentence = ' '.join( reverse_index.get(i-3, '*') for i in sequence)
    print(sentence)

changeSentence(0)
changeSentence(1)

#train_data : 시퀀스의 배열   [3,42,1,2,3]
#원핫인코딩 - 내부함수 말고 한번 만들어보자 
#10000개 까지만 불러오기로 함
import numpy as np  
def vectorize_sentences(sequences, dimensions=10000):
    results = np.zeros( (len(sequences), dimensions) ) #zeros 가 매개변수로 tuple을 받아간다  
    #문장개수 * 10000 개의 2차원배열, 0으로 채우는 
    #0,0,0,0,0,0,,,,,,,,,,,0 
    #0,0,0,0,0,0,,,,,,,,,,,0
    for i, sequence in enumerate(sequences):#enumerate(list타입) - 인덱스와 요소를 하나씩 전달한다
        #[1,4,11,12,6,5,4]) #해당위치를 1로 채운다 
        results[i, sequence] = 1.
        #넘파이배열 a[ [1,2,3,7,8] ] = 1
    return results 

X_train = vectorize_sentences(train_data) #시퀀스 =>원핫인코딩 
X_test = vectorize_sentences(test_data)
print(X_train[:3])

#훈련셋과 검증셋을 나눈다 - 전체 데이터 25000
X_val = X_train[:10000]#검증 : 10000
X_train = X_train[10000:] #훈련셋은 15000개
y_val = train_labels[:10000]
y_train = train_labels[10000:]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,  activation='sigmoid'))

model.compile( optimizer='rmsprop', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_val, y_val))
#평가하기 
print("훈련셋", model.evaluate(X_train, y_train))
print("테스트셋", model.evaluate(X_test, test_labels))

#예측하기 
pred = model.predict(X_test)
print(pred[:10]) #라벨이 1이 될 확률을 준다 
def changeData(pred):
    for i in range(len(pred)):
        if pred[i] <0.5:
            pred[i] = 0 
        else:
            pred[i] = 1
    return pred

pred = changeData(pred)
for i in range(0, 40):
    print(pred[i], test_labels[i])


# #훈련과 검증  정확도 그리기 
import matplotlib.pyplot as plt 
history_dict = history.history 
acc = history_dict['accuracy']  #훈련셋 정확도
val_acc = history_dict['val_accuracy'] #검증셋 정확도 

length = range(1,  len(history_dict['accuracy'])+1 ) #x축만들기 
plt.plot( length, acc , 'bo',  label='Training acc')
plt.plot( length, val_acc , 'b', label='Validation acc')
plt.title("Training and Validation acc")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

