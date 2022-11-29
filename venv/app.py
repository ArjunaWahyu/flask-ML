import os
from pyexpat import model
import numpy as np
import flask
import pickle
from flask import Flask, redirect, url_for, request, render_template


# creating instance of the class
app = Flask(__name__, template_folder='templates')

# to tell flask what url should trigger the function index()

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 2)
    loaded_model = pickle.load(open("./model/model.pkl", "rb"))  # load the model
    # predict the values using loded model
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        name = request.form['name']
        reading_score = request.form['reading_score']
        writing_score = request.form['writing_score']

        to_predict_list = list(map(float, [reading_score, writing_score]))
        result = ValuePredictor(to_predict_list)

        if float(result) == 0:
            prediction = 'Congrats, You Pass The Test :D'
        elif float(result) == 1:
            prediction = 'Your test isn\'t passed'
        elif float(result) == 2:
            prediction = 'Your test is passed'

        return render_template("result.html", prediction=prediction, name=name)


if __name__ == "__main__":
    app.run(debug=False)  # use debug = False for jupyter notebook