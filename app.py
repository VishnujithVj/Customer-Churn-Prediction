# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:54:46 2024

@author: vishnu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your dataset and model
df_1 = pd.read_csv("C:/Users/LENOVO/OneDrive/Desktop/customer churn/New_churn.csv")

# Load the trained model
model = pickle.load(open("C:/Users/LENOVO/OneDrive/Desktop/customer churn/model.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Collect form data dynamically
    input_data = [request.form[f'query{i}'] for i in range(1, 20)]
    
    # Create a DataFrame with the user input data
    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure'
    ])
    
    # Combine the new data with the original dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    
    # Drop the 'tenure' column
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    # Convert categorical variables into dummy/indicator variables
    df_2_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                         'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
    
    # Predict the churn probability
    single = model.predict(df_2_dummies.tail(1))
    probability = model.predict_proba(df_2_dummies.tail(1))[:, 1]
    
    # Determine the prediction message
    if single == 1:
        o1 = "This customer is likely to be churned!!"
    else:
        o1 = "This customer is likely to continue!!"
    o2 = f"Confidence: {probability[0] * 100:.2f}%"
    
    return render_template('home.html', output1=o1, output2=o2, **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)
