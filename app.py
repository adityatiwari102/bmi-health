import flask
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
filename = 'bmi_health_model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')

def main():
    return render_template('home.html')

@app.route('/predict', methods= ['POST'])

def home():
    age = request.form['a']
    height = request.form['b']
    weight = request.form['c']

    arr = np.array([[age, height, weight]])
    pred = model.predict(arr)

    return render_template('predict.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

