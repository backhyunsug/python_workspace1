import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier #중요도파악 
from sklearn.preprocessing import StandardScaler 

import os #파일이나 폴더경로를 정확하기 지정하려고 
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

#1.불필요한 열 삭제 
print(train.head)
