from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = pickle.load(open('model/logistic_regression_model.pkl','rb'))
scaler = pickle.load(open('model/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])

def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    
    input_data = [np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal])]
    std_data = scaler.transform(input_data)
    # Make predictions using the loaded model
    pred = model.predict(std_data)

    if pred[0] == 0:
        if sex == 0:
            result_text = "She does not have heart disease."
        else:
            result_text = "He does not have heart disease."
    else:
    # else:
        if sex == 0:
            result_text = "She has heart disease."
        else:
            result_text = "He has heart disease."
    # else: result_text = ""

    return render_template('index.html', prediction_text=result_text)

@app.route('/clear')

def clear():
    global result_text
    result_text = ""
    return render_template('index.html', prediction_text=result_text)

if __name__ == '__main__':
    app.run(debug=True)