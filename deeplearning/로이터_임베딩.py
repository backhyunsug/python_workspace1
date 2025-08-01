#중요: konlpy하고 Korpora 는 pip 로 설치해야 한다 
#https://fasttext.cc/docs/en/crawl-vectors.html
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from konlpy.tag import Okt
import re
import os, pathlib, shutil, random
import numpy as np # Used for dummy dataset creation
from keras import layers
import requests
from gensim.models import KeyedVectors

#https://ko-nlp.github.io/Korpora/ko-docs/corpuslist/nsmc.html




def create_dataset(base_dir="reuter"):
	if os.path.exists(base_dir):
		try:
			shutil.rmtree(base_dir) 
			print(f"Removed existing dummy dataset at {base_dir}")
		except OSError as e:
			print(f"Error removing directory {base_dir}: {e}")
			pass

	i=0
	with open("./data/r8-train-all-terms.txt","r") as f:
		for line in f:
			label, sentence = line.split("\t")
			i=i+1
			#if i>=5: 
			#	break 

			#디렉토리만들기
			print(os.path.join(base_dir, label))

			os.makedirs(os.path.join(base_dir, label), exist_ok=True)
			with open(os.path.join(base_dir, label, f"{label}_{i}.txt"), "w", encoding="utf-8") as f:		f.write(sentence)
			print(i)
					
	with open("./data/r8-test-all-terms.txt","r") as f:
		for line in f:
			label, sentence = line.split("\t")
			i=i+1
			#if i>=5: 
			#	break 

			#디렉토리만들기
			print(os.path.join(base_dir, label))

			os.makedirs(os.path.join(base_dir, label), exist_ok=True)
			with open(os.path.join(base_dir, label, f"{label}_{i}.txt"), "w", encoding="utf-8") as f:		f.write(sentence)
			print(i)
	print(f"{i}개 작성")

#create_dataset()
#train/test/valid 3개로 쪼개기 
#데이터 전체 개수 : 7674개  

def loadFile(base_dir="reuter"):
	labelnames = os.listdir(base_dir)
	print(labelnames)
	total_cnt = 0
	for label in labelnames:
		cnt = len(os.listdir(base_dir+"/"+label))
		total_cnt += cnt 
		print(label, cnt)
	print("전체 개수 : ", total_cnt)

loadFile()

"""
acq 2292
crude 374
earn 3923
grain 51
interest 271
money-fx 293
ship 144
trade 326
전체 개수 :  7674
"""

import requests
import subprocess
import re
import string
import tensorflow as tf
from keras import models, layers  
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import os, pathlib, shutil, random
import numpy as np
import pandas as pd # pandas added for the dummy example, can be removed if not used elsewhere
# You'll likely need to install gensim: pip install gensim
from gensim.models import KeyedVectors



batch_size=32
train_ds = keras.utils.text_dataset_from_directory(
    "reuter/", batch_size=batch_size
)

text_only_train_ds = train_ds.map(lambda x, y: x)


# --- Word2Vec specific changes start here ---

# Define the path to your pre-trained Word2Vec file
# You need to download GoogleNews-vectors-negative300.bin.gz or similar.
# For example: https://code.google.com/archive/p/word2vec/
# And then decompress it.
path_to_word2vec_file = "GoogleNews-vectors-negative300.bin" # <<< Make sure this path is correct!

# Load the Word2Vec model using gensim
try:
    word2vec_model = KeyedVectors.load_word2vec_format(path_to_word2vec_file, binary=True)
    print(f"Word2Vec model loaded. Vector size: {word2vec_model.vector_size}")
    embedding_dim = word2vec_model.vector_size # Set embedding_dim to match Word2Vec
except FileNotFoundError:
    print("You can usually find it by searching for 'GoogleNews-vectors-negative300.bin.gz' and decompressing it.")
    exit() # Exit if the model isn't found, as the rest of the code depends on it.
except Exception as e:
    print(f"An error occurred while loading Word2Vec model: {e}")
    exit()

max_tokens = 20000 # Same as before
#시퀀스로 변경하기 
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=600, # Use a fixed sequence length, not embedding_dim
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE) # Use AUTOTUNE for better performance

vocabulary = text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

# Initialize embedding_matrix with zeros
embedding_matrix = np.zeros((max_tokens, embedding_dim))

# Populate embedding_matrix with Word2Vec vectors
print("Populating embedding matrix with Word2Vec vectors...")
hits = 0
misses = 0
for word, i in word_index.items():
    if i < max_tokens:
        try:
            embedding_vector = word2vec_model[word] # Access vector from gensim KeyedVectors
            embedding_matrix[i] = embedding_vector
            hits += 1
        except KeyError: # Word not found in Word2Vec vocabulary
            misses += 1
# Optionally, handle out-of-vocabulary words more gracefully, e.g., with random initialization or average vector.
print(f"Converted {hits} words ({misses} misses)")


embedding_layer = layers.Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False, # Keep embeddings fixed (pre-trained)
    mask_zero=True, # Important for variable-length sequences with padding
)

# --- Word2Vec specific changes end here ---


# Build the model
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(8, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("word2vec_embeddings_sequence_model2.keras",
                                    save_best_only=True)
]
print("\nStarting model training with Word2Vec embeddings...")
model.fit(int_train_ds, epochs=10, callbacks=callbacks)

print("\nLoading the best model and evaluating on test set...")
model = keras.models.load_model("word2vec_embeddings_sequence_model2.keras")











