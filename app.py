from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl','rb'))

# Create a Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    age = int(request.form['age'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    glucose = float(request.form['glucose'])
    heart_rate = int(request.form['heart_rate'])
    cholesterol = int(request.form['cholesterol'])
    
    # Create a numpy array with the user input values
    input_data = np.array([[age, systolic_bp, diastolic_bp, glucose, heart_rate, cholesterol]])
    
    # Use the trained model to make a prediction
    prediction = model.predict(input_data)
    
    # Return the predicted risk level to the user
    return render_template('index.html', prediction_text='The predicted risk level is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)




