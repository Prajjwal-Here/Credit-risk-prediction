from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('models/final_rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    person_age = float(request.form['person_age'])  # User's age
    person_income = float(request.form['person_income'])  # User's annual income
    loan_amnt = float(request.form['loan_amnt'])  # The loan amount user is applying for
    loan_int_rate = float(request.form['loan_int_rate'])  # Loan interest rate
    
    # Handle categorical variables (one-hot encoding)
    person_home_ownership = request.form['person_home_ownership']
    if person_home_ownership == "OWN":
        person_home_ownership_OWN = 1
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_RENT = 0
    elif person_home_ownership == "MORTGAGE":
        person_home_ownership_OWN = 0
        person_home_ownership_MORTGAGE = 1
        person_home_ownership_RENT = 0
    else:
        person_home_ownership_OWN = 0
        person_home_ownership_MORTGAGE = 0
        person_home_ownership_RENT = 1

    # Loan grade one-hot encoding
    loan_grade_A = 1 if request.form['loan_grade'] == "A" else 0
    loan_grade_B = 1 if request.form['loan_grade'] == "B" else 0
    loan_grade_C = 1 if request.form['loan_grade'] == "C" else 0
    loan_grade_D = 1 if request.form['loan_grade'] == "D" else 0
    loan_grade_E = 1 if request.form['loan_grade'] == "E" else 0
    loan_grade_F = 1 if request.form['loan_grade'] == "F" else 0
    loan_grade_G = 1 if request.form['loan_grade'] == "G" else 0
    
    # Prepare the features list to match the model's expectations (23 features)
    features = [
        person_age, 
        person_income, 
        loan_amnt, 
        loan_int_rate,
        person_home_ownership_OWN, 
        person_home_ownership_MORTGAGE, 
        person_home_ownership_RENT,
        loan_grade_A, loan_grade_B, loan_grade_C, loan_grade_D, loan_grade_E, loan_grade_F, loan_grade_G
        # Add additional features if needed (e.g., one-hot encoded columns for other categorical data)
    ]
    
    # Prediction using the trained model
    prediction = model.predict([np.array(features).reshape(1, -1)])

    # Return prediction result
    result = "Default" if prediction[0] == 1 else "No Default"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
