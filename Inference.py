from flask import Flask
from flask import request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
# model_path = "C:/Users/DELL/Documents/Jupyter notebooks/Flask/reg_model"
# model = pickle.load(open(model_path, 'rb'))
file_name = "reg_model"
model = pickle.load(open(file_name, 'rb'))


@app.route('/')
def home():
    return "Insert characteristics of your country!"


@app.route('/predict')
def predict():
    data = np.array([[float(request.args.get('GDP')),float(request.args.get('social_support')),float(request.args.get('Life_expectancy')),
                      float(request.args.get('Freedom')),float(request.args.get('Generosity')),float(request.args.get('corruption'))]])
    y_pred = model.predict(data).tolist()[0]
    return jsonify(y_pred)


@app.route('/predict_many')
def predict_many():
    data = request.get_json(force=True)
    data = pd.DataFrame(json.loads(data))
    pred = model.predict(data).tolist()
    pred_to_client = [f"# query {key+1} score is: {value}" for key, value in enumerate(pred)]
    return jsonify(pred_to_client)


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
