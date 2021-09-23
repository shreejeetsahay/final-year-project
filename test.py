from keras.models import model_from_json
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import keras.backend as K
import os
import json
import sys
import re

MAX_SEQUENCE_LENGTH=30

json_file = open('classificationlstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# model.summary()
model.load_weights('classificationlstm.hdf5')

with open('conjunction.txt','r') as fp:
	conjunctions=fp.read().split(",")


message="i order you to go in the forward direction 10 cm and turn left by 15 cm"
conjlist=[]
conjFlg=False
for i in conjunctions:
	x=message.find(i)
	
	if x!=-1:
		message=message.split(i)
		conjFlg=True
		break

if not conjFlg:
	message=[message]



commands=['Null','forward','backward','right','left','stop']
with open('dictionarylstm.json', 'r') as dictionary_file:
	word_index = json.load(dictionary_file)

noutput=[]
for i in message:
    digit=re.findall(r'\d+',i)  
    if "cm" in i or "centimeter" in i or "centimeters" in i or "cms" in i:
        distance="cm"
    elif "m" in i or "meter" in i or "meters" in i:
        distance="m"
    else:
        distance=""
    #print(i)
    test = []
    data = []
    for word in word_tokenize(i):
        if word in word_index:
            test.append(word_index[word])
        else:
            pass
    data = sequence.pad_sequences([test], maxlen=MAX_SEQUENCE_LENGTH)

    output=model.predict_classes([data])
    print(output)


    if output[0]:
        noutput.append(f"{commands[output[0]]} {digit} , {distance}")
print(noutput)