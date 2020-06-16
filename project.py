
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask import jsonify
from tensorflow.keras.models import load_model


from tensorflow.keras.preprocessing import image

import numpy as np

from diseases import * 

app = Flask(__name__);
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def init():

    #define global variables
    global model
    global lists
    global remedies
    #import model
    model = load_model('../cnn/AlexNetModel.hdf5')
    
    lists = load_diseases()
    remedies = load_remedies()

   


@app.route("/", methods=['POST'])
def index():
    new_image  =  request.files['file'].stream
    new_img = image.load_img(new_image, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)

    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
        if item == j:
            class_name = lists[index]
            remedy = remedies[index]
    response = class_name.split("___")
    return jsonify(
            plant=response[0],
            status=response[1],
            remedy=remedy
                );


if __name__ == '__main__':
    init()
    app.run()

