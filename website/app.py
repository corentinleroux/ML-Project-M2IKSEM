from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import joblib
import pickle
import numpy as np
from IPython.display import display
import sys
import random 

# Array of print value 
divorce = ["You should have listen to your mom. Divorce", "Sorry to say but there is no chance for you. Divorce","You should break up (and no, it's not your partner's fault)","It's not too late to find the right partner. Divorce as soon as possible.","How could you marry this person? Divorce now!"]
not_divorce = ["You should stay together", "All fine, stay together", "How dare you have any doubts","It is the partner of your life!","You are in perfect love!"]

app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug = True)

@app.route("/", methods=["GET","POST"])
def home():
    
    # If a form is submitted
    if request.method == "POST":
        # Take the data from the form 
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1, 54)
        display(to_predict)

        # Loading the model 
        loaded_model = joblib.load(open("model/bagging_extratrees.pkl", 'rb')) # others models can be find in the model folder

        # Making prediction 
        result = loaded_model.predict(to_predict)[0]
        
        # print the output (either 0 or 1)
        print(result) 

        # Change the HTML value of {{ output }} field
        if result == 1 or result =="1" :
          prediction = random.choice(divorce)
        else :
          prediction = random.choice(not_divorce)

    # If no form is submitted, leave {{ output }} field empty  
    else:
        prediction = ""
    
    # HTML model     
    return render_template("home.html", output = prediction)