import numpy as np
from PIL import Image
import torch
from NetLearning import Net
import os
from flask_cors import CORS
from flask import Flask, request, render_template, jsonify, url_for
from datetime import datetime
import base64
import json


app = Flask(__name__)
CORS(app, headers=['Content-Type'])


def load_dict(PATH="model/best_model.npy"):
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.train()
    return model


model = load_dict()



class Predict:
    def __init__(self, model, image_path=None, image=None):
        self.net = model
        if image_path == None:
            self.image = image
        else:
            self.image_path = image_path

    def load_image(self):  # Возвращает матрицу пикселей картинки
        if self.image_path == None:
            img = Image.open(self.image)
        else:
            img = Image.open(self.image_path)
        img.thumbnail((64, 64))
        return np.asarray(img)

    def prediction(self, pic_size=64*64*4):
        dictionary = {0: 'hyperbola', 1: 'sigmoid', 2: 'abs', 3: 'linear', 4: 'parabola'}
        test = self.load_image()
        test = (torch.Tensor(test - np.mean(test)) / np.abs(np.std(test)))
        model = self.net
        mass = np.array(model(test.view(-1, pic_size)).tolist())
        probability = np.max(np.round(np.power(np.e, mass) / np.sum(np.power(np.e, mass)), 3))
        return dictionary[model(test.view(-1, pic_size)).argmax().tolist()], probability



@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
	return render_template('index.html')


@app.route("/draw_preds", methods=['POST'])
def draw_prediction(network=model):
    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        prediction, _ = Predict(model=network, image=image)

        return json.dumps(prediction)



@app.route("/predict", methods=["POST"])
def predict(network=model):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "something went wrong!"

        user_file = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."

        else:
            path = os.path.join("C:\\server" + '\\static\\' + user_file.filename)
            user_file.save(path)
            classes, conf = Predict(model=network, image_path=path).prediction()
            return jsonify({
                "status": "success",
                "prediction": classes,
                "confidence": str(conf),
                "upload_time": datetime.now()
            })


port = int(os.environ.get("PORT", 4016))
app.run(host='192.168.0.11', port=port)
