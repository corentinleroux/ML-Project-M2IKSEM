from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import joblib
import pickle
import numpy as np
from IPython.display import display
import sys

app = Flask(__name__)
if __name__ == '__main__':
    app.run(debug = True)

@app.route("/", methods=["GET","POST"])
def home():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("boosting.pkl")
        '''
        
        # Get values through input bars
        q1 = request.form.get("Q1")
        q2 = request.form.get("Q2")
        q3 = request.form.get("Q3")
        q4 = request.form.get("Q4")
        q5 = request.form.get("Q5")
        q6 = request.form.get("Q6")
        q7 = request.form.get("Q7")
        q8 = request.form.get("Q8")
        q9 = request.form.get("Q9")
        q10 = request.form.get("Q10")
        q11 = request.form.get("Q11")
        q12 = request.form.get("Q12")
        q13 = request.form.get("Q13")
        q14 = request.form.get("Q14")
        q15 = request.form.get("Q15")
        q16 = request.form.get("Q16")
        q17 = request.form.get("Q17")
        q18 = request.form.get("Q18")
        q19 = request.form.get("Q19")
        q20 = request.form.get("Q20")
        q21 = request.form.get("Q21")
        q22 = request.form.get("Q22")
        q23 = request.form.get("Q23")
        q24 = request.form.get("Q24")
        q25 = request.form.get("Q25")
        q26 = request.form.get("Q26")
        q27 = request.form.get("Q27")
        q28 = request.form.get("Q28")
        q29 = request.form.get("Q29")
        q30 = request.form.get("Q30")
        q31 = request.form.get("Q31")
        q32 = request.form.get("Q32")
        q33 = request.form.get("Q33")
        q34 = request.form.get("Q34")
        q35 = request.form.get("q35")
        q36 = request.form.get("Q36")
        q37 = request.form.get("Q37")
        q38 = request.form.get("Q38")
        q39 = request.form.get("Q39")
        q40 = request.form.get("Q40")
        q41 = request.form.get("Q41")
        q42 = request.form.get("Q42")
        q43 = request.form.get("Q43")
        q44 = request.form.get("Q44")
        q45 = request.form.get("Q45")
        q46 = request.form.get("Q46")
        q47 = request.form.get("Q47")
        q48 = request.form.get("Q48")
        q49 = request.form.get("Q49")
        q50 = request.form.get("Q50")
        q51 = request.form.get("Q51")
        q52 = request.form.get("Q52")
        q53 = request.form.get("Q53")
        q54 = request.form.get("Q54")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52,q53,q54]], columns = ["Sorry_end", "Ignore_diff", "begin_correct", "Contact", "Special_time","No_home_time", "2_strangers", "enjoy_holiday", "enjoy_travel","common_goals", "harmony", "freeom_value", "entertain", "people_goals","dreams", "love", "happy", "marriage", "roles", "trust", "likes","care_sick", "fav_food", "stresses", "inner_world", "anxieties","current_stress", "hopes_wishes", "know_well", "friends_social","Aggro_argue", "Always_never", "negative_personality","offensive_expressions", "insult", "humiliate", "not_calm","hate_subjects", "sudden_discussion", "idk_what's_going_on","calm_breaks", "argue_then_leave", "silent_for_calm", "good_to_leave_home", "silence_instead_of_discussion","silence_for_harm", "silence_fear_anger", "I'm_right", "accusations","I'm_not_guilty", "I'm_not_wrong", "no_hesitancy_inadequate","you're_inadequate", "incompetence"])
        display(X)     
        # Get prediction
        prediction = clf.predict(X)[0]
        '''
       # prediction = "yes"
        
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1, 54)
        display(to_predict)
        loaded_model = joblib.load(open("rf_clf_bagging2.pkl", 'rb'))
        result = loaded_model.predict(to_predict)
        print(result)
        if result == 1 or result =="1" :
          prediction = "You should divorce"
        else :
          prediction = "All fine, you can stay together"
        
    else:
        prediction = ""
        
    return render_template("home.html", output = prediction)