import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle


def load_model():
    file_name = "models\model_file.p"
    with open(file_name, "rb") as pickled:
        data = pickle.load(pickled)
        model = data["model"]
    return model

app = Flask(__name__)
@app.route('/predict', methods=['GET'])
def predict():
    # input features
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1,-1)
    # load model
    model = load_model()
    prediction = model.predict(x_in)[0]
    # return prediction
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == '__main__':
    application.run(debug=True)
    
# predict()
