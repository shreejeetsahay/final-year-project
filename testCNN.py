from keras.models import model_from_json
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import keras.backend as K
import os
import json
import sys
import re

message="stop now "

MAX_SEQUENCE_LENGTH=30

json_file = open('classification.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# model.summary()
model.load_weights('classification.hdf5')

with open('conjunction.txt','r') as fp:
	conjunctions=fp.read().split(",")




commands=['Null','forward','backward','right','left','stop']
with open('dictionary.json', 'r') as dictionary_file:
	word_index = json.load(dictionary_file)

test = []
data = []
for word in word_tokenize(message):
	if word in word_index:
		test.append(word_index[word])
	else:
		pass
data = sequence.pad_sequences([test], maxlen=MAX_SEQUENCE_LENGTH)

output=model.predict_classes([data])

print(f"{commands[output[0]]}")
