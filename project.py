
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask import jsonify
from tensorflow.keras.models import load_model


from tensorflow.keras.preprocessing import image

import numpy as np


app = Flask(__name__);
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def init():

    #define global variables
    global model
    global lists
    #import model
    model = load_model('../cnn/AlexNetModel.hdf5')
    lists = ['Apple___Apple scab', 'Apple___Black rot', 'Apple___Cedar apple rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry(including sour)___Powdery mildew', 'Cherry(including sour)___healthy', 'Corn (maize)___Cercospora leaf spot Gray leaf spot', 'Corn (maize)___Common rust ', 'Corn (maize)___Northern Leaf Blight', 'Corn (maize)___healthy', 'Grape___Black rot', 'Grape___Esca (Black Measles)', 'Grape___Leaf blight (Isariopsis Leaf Spot)', 'Grape___healthy', 'Orange___Haunglongbing (Citrus greening)', 'Peach___Bacterial spot', 'Peach___healthy', 'Pepper bell___Bacterial spot', 'Pepper ___healthy', 'Potato___Early blight', 'Potato___Late blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early blight', 'Tomato___Late blight', 'Tomato___Leaf Mold', 'Tomato___Septoria leaf spot', 'Tomato___Spider mites Two spotted spider mite', 'Tomato___Target Spot', 'Tomato___Tomato Yellow Leaf Curl Virus', 'Tomato___Tomato mosaic virus', 'Tomato___healthy']



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
    response = class_name.split("___")
    return jsonify(
            plant=response[0],
            status=response[1]
                );


if __name__ == '__main__':
    init()
    app.run()

