from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import joblib
import pickle
import numpy as np
from IPython.display import display
import sys
import random 

divorce = ["You should have listen to your mom. Divorce", "Sorry to say but their is no chance for you. Divorce"]
not_divorce = ["You should stay together", "All fine, stay together", "How dare you have any doubts"]

app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug = True)

@app.route("/", methods=["GET","POST"])
def home():
    
    # If a form is submitted
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1, 54)
        display(to_predict)
        loaded_model = joblib.load(open("model/bagging_extratrees.pkl", 'rb'))
        result = loaded_model.predict(to_predict)[0]
        print(result)
        if result == 1 or result =="1" :
          prediction = random.choice(divorce)
        else :
          prediction = random.choice(not_divorce)
        
    else:
        prediction = ""
        
    return render_template("home.html", output = prediction)