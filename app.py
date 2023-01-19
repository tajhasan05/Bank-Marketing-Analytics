import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the Model
pickled_model=pickle.load(open('xgbmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['post'])
def predict_api():
    data=request.json['data']
    print(data)
    output=pickled_model.score(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)
