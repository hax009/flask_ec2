from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Load the trained model
model_path = '/home/ec2-user/flask_app/model_rf.pkl' 
with open(model_path, 'rb') as model_file:
    best_model_rf = pickle.load(model_file)

# Load the trained scaler
scaler_path = '/home/ec2-user/flask_app/scaler.pkl'  
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)



# Define the list of feature names used during training
features = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','gender_Female','gender_Male','Partner_No','Partner_Yes',
                'Dependents_No','Dependents_Yes','PhoneService_No','PhoneService_Yes','MultipleLines_No','MultipleLines_No phone service',
                'MultipleLines_Yes','InternetService_DSL','InternetService_Fiber optic','InternetService_No','OnlineSecurity_No','OnlineSecurity_No internet service',
                'OnlineSecurity_Yes','OnlineBackup_No','OnlineBackup_No internet service','OnlineBackup_Yes','DeviceProtection_No','DeviceProtection_No internet service',
                'DeviceProtection_Yes','TechSupport_No','TechSupport_No internet service','TechSupport_Yes','StreamingTV_No','StreamingTV_No internet service',
                'StreamingTV_Yes','StreamingMovies_No','StreamingMovies_No internet service','StreamingMovies_Yes','Contract_Month-to-month',
                'Contract_One year','Contract_Two year','PaperlessBilling_No','PaperlessBilling_Yes','PaymentMethod_Bank transfer (automatic)',
                'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check']





# Route to render the HTML form
@app.route('/')
def index():
    return render_template('test_23.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_churn():
    if request.method == 'POST':
        # Retrieve form data
        customer_id = request.form['customerID']
        gender = request.form['gender']
        senior_citizen = int(request.form['seniorCitizen'])
        partner = request.form['partner']
        dependents = request.form['dependents']
        tenure = int(request.form['tenure'])
        phone_service = request.form['phoneService']
        multiple_lines = request.form['multipleLines']
        internet_service = request.form['internetService']
        online_security = request.form['onlineSecurity']
        online_backup = request.form['onlineBackup']
        device_protection = request.form['deviceProtection']
        tech_support = request.form['techSupport']
        streaming_tv = request.form['streamingTV']
        streaming_movies = request.form['streamingMovies']
        contract = request.form['contract']
        paperless_billing = request.form['paperlessBilling']
        payment_method = request.form['paymentMethod']
        monthly_charges = float(request.form['monthlyCharges'])
        total_charges = float(request.form['totalCharges'])
        
        # Create a dictionary with form data
        new_customer_data = {
            'customerID': customer_id,
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Convert dictionary to DataFrame
        new_customer_df = pd.DataFrame([new_customer_data])

  

        # Apply the same preprocessing steps
        new_customer_df['TotalCharges'] = pd.to_numeric(new_customer_df['TotalCharges'], errors='coerce')
        df_dummies = pd.get_dummies(new_customer_df.drop('customerID', axis=1))




         # Ensure columns match exactly
        df_dummies = df_dummies.reindex(columns=features, fill_value=0)
        new_customer_data_scaled = scaler.transform(df_dummies)


        # Predict the probabilities for churn and no churn for the new customer
        predicted_probabilities = best_model_rf.predict_proba(new_customer_data_scaled)
        churn_probability = predicted_probabilities[0][1]

        return render_template('test_23.html', churn_probability=churn_probability)

    return render_template('test_23.html')

   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

