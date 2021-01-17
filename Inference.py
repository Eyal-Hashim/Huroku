from flask import Flask
from flask import request, jsonify
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model_path = "C:/Users/DELL/Documents/Jupyter notebooks/Flask/reg_model"
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def home():
    return "Insert characteristics of your country!"


@app.route('/predict')
def predict():
    data = np.array([[request.args.get('GDP'),request.args.get('social_support'),request.args.get('Life_expectancy'),
                      request.args.get('Freedom'),request.args.get('Generosity'),request.args.get('corruption')]])
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
    app.run(debug=True)