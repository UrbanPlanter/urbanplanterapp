import os
import sys
import pandas as pd
# Flask
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

# Declare class names for labels
class_dict = {'Begonia Maculata':0, 'Coleus':1, 'Elephant Ear':2, 'House Leek':3, 'Jade Plant':4, 'Lucky Bamboo':5, 'Moon Cactus':6, 'Nerve Plant':7, 'Paddle Plant':8, 'Parlor Palm':9, 'Poinsettia':10, 'Sansevieria Ballyi':11, 'String Of Banana':12, 'Venus Fly Trap':13, 'Zebra Plant':14}
class_names = list(class_dict.keys())

# Model saved with Keras model.save()
MODEL_PATH_1 = './models/UrbanPlantClassifier.h5'

# Load my own trained model
model_1 = load_model(MODEL_PATH_1)

print('Model loaded.. Check http://127.0.0.1:5000/')
print('Model loaded. Start serving...')

def model_predict1(img, model):
    # Preprocessing the image
    img = img.resize((256, 256))
    x = image.img_to_array(img) / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds1 = model_predict1(img, model_1)
        index_pred1 = np.argmax(preds1)

        # Confidence
        confidence1 = "{}".format(preds1[0][index_pred1])
        index = 0
        result = class_names[index_pred1]
        index = index_pred1
        confidence = float(confidence1)*100

        # Read description from excel file according to the prediction of given image
        data = pd.read_excel (r'.\urbanplanterapp\descriptions.xlsx') 
        df = pd.DataFrame(data, columns= ['Label','Class','Scientific name','Genus', 'Habitat'])
        row = df.iloc[index]
        myDes = []
        myDes.append(round(confidence,2))
        myDes.extend([row[1],row[2],row[3],row[4]])

        # Return predicted class, confidence and description.
        return jsonify(result=[result, 'Class:\t'+myDes[1], 'Confidence:\t'+str(myDes[0])+'%', 'Scientific name:\t'+myDes[2], 'Genus:\t'+myDes[3], 'Habitat:\t'+myDes[4]])
    return None

if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
