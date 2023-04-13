
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras import backend
from tensorflow import keras
import tensorflow as tf

# global graph
from skimage.transform import resize

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model(r'..\models\garbage1.h5')
       # Necessary
# print('Model loaded. Start serving...')

@app.route('/',methods=['POST','GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['image']

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'predictions',f.filename)
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        a=np.argmax(model.predict(x),axis=1)
        
        
       # preds = model.predict_classes(x)
        index = ['cardboard','glass','metal','paper','plastic','trash']
        text = "The Predicted Garbage is : "+str(index[a[0]])
        
               # ImageNet 
        
        return text
    
if __name__ == '__main__':
    app.run(debug=False,threaded = False)


