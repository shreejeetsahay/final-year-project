from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from keras.models import model_from_json
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import keras.backend as K
import os
import json
import sys
import re
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    name = TextField('Enter Text:', validators=[validators.required()])

    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)

        print(form.errors)
        if request.method == 'POST':
            name=request.form['name']

        if form.validate():
        # Save the comment here.
            print(name)
            flash(predict(name))
        else:
            flash('All the form fields are required. ')

        return render_template('hello.html', form=form)

def predict(message):
    # print(message)
    MAX_SEQUENCE_LENGTH=30

    json_file = open('classificationlstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # model.summary()
    model.load_weights('classificationlstm.hdf5')
    
    with open('conjunction.txt','r') as fp:
        conjunctions=fp.read().split(",")

    conjFlg=False
    #message="i order you to go in the forward direction for 10 cm and go left for 20 cm"
    conjlist=[]
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
            noutput.append(f"{commands[output[0]]},{digit} , {distance}")

    K.clear_session()
    return str(noutput)

if __name__ == "__main__":

    app.run()
