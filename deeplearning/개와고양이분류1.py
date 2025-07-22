#폴더에 train, test, validation 등으로 나눈다음에 각자에 폴더에 라벨을 만들고 
#데이터 넣어놓고, ImageDataGenerator 나 DataSet 을 통해서 파일을 직접 읽어서 
#학습한다. 데이터 증강  ImageDataGenerator(초창기부터)-폴더로부터 직접 이미지 파일
#을 읽어서 각종 처리를 해서 원하는 만큼 데이터를 늘려서 가져온다.
#좀더 정밀하게 비슷한 일을 한다. DataSet - Tensor 2.xx이후 추가 
#이미지 => numpy배열로 바꿔서 학습:데이터가 충분히 많으면 

#데이터셋이 작을때 이미 학습된 CNN하고 작업할때 쓸 예정임 
#1.cats_and_dogs_small - train, test, validation 만들어서 
#  train/cat- 고양이사진 1000장 dog-개사진 1000장 
#  test/cat- 고양이사진 500장 dog-개사진 500장
#  validation/cat- 고양이사진 500장 dog-개사진 500장

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 
import random 
import PIL.Image as pilimg 
import imghdr
import pandas as pd 
import pickle 
import keras 
import os 
import shutil #디렉토리 만들기, 삭제등을 담당하는 라이브러리

#원본데이터셋이 있는 위치 경로 
original_dataset_dir = "./data/cats_and_dogs/train"

#옮길위치 - 기본 폴더 
base_dir = "./data/cats_and_dogs_small"

train_dir = os.path.join(base_dir, 'train')
test_dir   = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

#ImageDataGenerator나 DataSet이나 둘 다 폴더보고 자동으로 라벨링을 한다 
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

def ImageCopy():
    #디렉토리내의 파일개수 알아내기
    totalCount = len(os.listdir(original_dataset_dir))
    print("전체개수", totalCount)

    #바복적인 실행을 위해서 디렉토리 삭제
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True, onerror=None) 

    #디렉토리 만들고
    os.makedirs(base_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(validation_dir)
    
    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)

    #파일 옮기기
    #옮길 파일명이 cat0.jpg, cat1.jpg ,,, cat1000.jpg 
    fnames = [ f'cat.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'cat.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'cat.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사
    
    fnames = [ f'dog.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'dog.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

    fnames = [ f'dog.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)  #1개씩 복사

ImageCopy()

