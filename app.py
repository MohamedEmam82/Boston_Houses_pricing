import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# starting point from where my app will run
app = Flask(__name__)   
# load linear regression model & scalar pickled files
LinRegModel = pickle.load(open('LinRegModel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))
# define first route / going to my home page / localhost
# by default once I hit this flask app, it'll redirect to home.html
@app.route('/')
def home():
    # I've to create home.html file inside templates folder
    return render_template('home.html')

# redirect to the predicting api application, using postman tool to send
# the request to the api then getting the output/prediction
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # make sure the input data is in json format
    data = request.json['data']
    print(data)
    # the data is in key-value format, must to reshaped into array of 1 row and
    # number of cols is the number of features
    # first transforming data.values() to a list
    data = list(data.values())
    # then transform & reshape the list into an array
    data = np.array(data).reshape(1,-1)
    print(data)
    # scaling the data 
    data = scalar.transform(data)
    # apply LinRegModel to the input data
    prediction = LinRegModel.predict(data)
    print(prediction[0]) # the output is 2d array
    # return the output into json format
    return jsonify(prediction[0])

# to run the application
if __name__ == '__main__':
    app.run(debug=True)