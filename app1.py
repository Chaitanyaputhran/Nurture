from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    # get the user input from the HTML form
    age = int(request.form['age'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    bmi = float(request.form['bmi'])
    heart_rate = int(request.form['heart_rate'])
    glucose = int(request.form['glucose'])

    # convert the user input into a numpy array
    input_data = np.array([[age, systolic_bp, diastolic_bp, bmi, heart_rate, glucose]])

    # use the random forest model to predict the risk level
    prediction = model.predict(input_data)

    # return the predicted risk level to the HTML page
    return render_template('result.html', prediction=prediction[0])
