import pandas
import os
import json
import numpy as np
from keras.models import Sequential
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
import sys
import keras.backend as K
from keras.utils import to_categorical

abc=pandas.read_csv("new_dataset.csv")

texts=abc.sentence
label=to_categorical(abc.command)


MAX_SEQUENCE_LENGTH = 30
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
dictionary = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

with open('dictionarylstm.json', 'w') as dictionary_file:
	json.dump(dictionary, dictionary_file)

test_x = data
#  target as categories
test_y = label

X_train, X_test, y_train, y_test = train_test_split(test_x, test_y, test_size=0.2,random_state=42)
print(X_train[0])
maxLen = 30
embeddingDim = 100

model = Sequential()
model.add(Embedding(len(word_index) + 1, embeddingDim, input_length=maxLen))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(LSTM(100))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16)
model = model
loss, accuracy = model.evaluate(X_test, y_test)

model_json = model.to_json()
with open('classificationlstm.json', 'w') as json_file:
	json_file.write(model_json)
model.save_weights('classificationlstm.hdf5')
print(accuracy)