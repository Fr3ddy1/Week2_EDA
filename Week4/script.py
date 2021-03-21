#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)
model = pickle.load(open('checkpoints/model_freddy.pkl','rb'))

#to tell flask what url shoud trigger the function index()
@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    '''
    DEFINE PREDICTION FUCTION
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)

    return render_template("index.html", prediction_text='The final decision of the University should be {}'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)
