from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
import joblib

app = Flask(__name__)

# Load your trained model and other necessary objects
model = joblib.load('model.pkl')  # Save your trained model to model.pkl
scaler = joblib.load('scaler.pkl')  # Save your scaler to scaler.pkl
poly = joblib.load('poly.pkl')  # Save your polynomial features to poly.pkl

@app.route('/')
def form():
    return render_template('credit_risk.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict(flat=True)
    
    # Convert form data to DataFrame
    input_data = pd.DataFrame([data], dtype=float)
    
    # Apply necessary transformations
    input_data['IncomePerPerson'] = input_data['MonthlyIncome'] / (input_data['NumberOfDependents'] + 1)
    input_data['DebtToIncomeRatio'] = input_data['DebtRatio'] * input_data['MonthlyIncome']
    
    # Handle GMM cluster assignment
    gmm = joblib.load('gmm.pkl')  # Load your GMM model
    gmm_labels = gmm.predict(input_data)
    input_data['GMM_Cluster'] = gmm_labels
    
    # Generate polynomial features
    poly_features = poly.transform(input_data)
    X_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(input_data.columns))
    
    # Scale features
    X_scaled = scaler.transform(X_poly)
    
    # Make prediction
    prediction = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)
    
    # Format the prediction probability to show only the first two digits
    #formatted_proba = np.round(prediction_proba, 2)

    output1='{0:.{1}f}'.format(prediction_proba[0][0], 2)
    output2='{0:.{1}f}'.format(prediction_proba[0][1], 2)
    
    # high>=0.66
    # mediam =>0.33
    # low <=0.1

    # return jsonify({'prediction': int(prediction[0])})

    if prediction == 0:
        return render_template('credit_risk.html',level="This Facility is in LOW Risk",pred1='The Probability of the non-default on this Facility is : {}'
                               .format(output1),pred2='The Probability of the default on this Facility is : {}'
                               .format(output2))
    else:
        return render_template('credit_risk.html',level="This Facility is in HIGH Risk",pred1='The Probability of the non-default on this Facility is : {}'
                               .format(output1),pred2='The Probability of the default on this Facility is : {}'
                               .format(output2))

if __name__ == '__main__':
    app.run(debug=True)
    