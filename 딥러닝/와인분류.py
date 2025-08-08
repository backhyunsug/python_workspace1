import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pickle

def loadData():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    # print(X)
    # print(y)
    # print(X.isna().sum())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

    return X_train, X_test, y_train, y_test

from keras import models, layers
def createModel():
    model = models.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
def preprocessing():
    X_train, X_test, y_train, y_test = loadData()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.fit_transform(X_test)

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

def main():
    X_train, X_test, y_train, y_test = preprocessing()
    model = createModel()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='와인분류_딥러닝.keras',
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    history = model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test),
                        callbacks=callbacks, batch_size=100)
    
    with open("와인분류_딥러닝.bin", "wb") as file:
        pickle.dump(history.history, file)

    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"훈련셋 손실값: {train_loss}, 정확도: {train_acc}")
    print(f"테스트셋 손실값: {test_loss}, 정확도: {test_acc}")

if __name__ == "__main__":
    # loadData()
    main()