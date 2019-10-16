import numpy as np
from PIL import Image
import torch
from NetLearning import Net
import os
from flask_cors import CORS
from flask import Flask, request, render_template, jsonify, url_for
from datetime import datetime
import base64
from Predict import Predict
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
            path = os.path.join(user_file.filename)
            classes, conf = Predict(model=network, image_path=path).prediction()
            return jsonify({
                "status": "success",
                "prediction": classes,
                "confidence": str(conf),
                "upload_time": datetime.now()
            })


port = int(os.environ.get("PORT", 4017))
app.run(host='192.168.0.11', port=port)
