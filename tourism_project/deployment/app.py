import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Praveen-kavedi/Tourism-Customer-prediction", filename="best_tourism_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Customer Purchase Prediction
st.title("Tourism Customer Purchase Prediction App")
st.write("The Tourism Customer Purchase Prediction App is an internal tool for 'Visit with Us' staff that predicts whether customers are likely to purchase the Wellness Tourism Package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the package.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact (method by which the customer was contacted)", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier (city category based on development)", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (duration of the sales pitch in minutes)", min_value=0, value=15)
Occupation = st.selectbox("Occupation (customer's occupation)", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender (gender of the customer)", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (total people accompanying the customer)", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Followups (total follow-ups by the salesperson)", min_value=0, value=3)
ProductPitched = st.selectbox("Product Pitched (type of product pitched to the customer)", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star (preferred hotel rating)", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status (marital status of the customer)", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips (average number of trips annually)", min_value=0, value=2)
Passport = st.selectbox("Passport (does the customer hold a valid passport?)", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (customer's satisfaction with the sales pitch)", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car (does the customer own a car?)", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (children below age 5 accompanying)", min_value=0, value=0)
Designation = st.selectbox("Designation (customer's designation in their organization)", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (gross monthly income of the customer)", min_value=0.0, value=20000.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'CityTier': CityTier,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase the Wellness Tourism Package" if prediction == 1 else "not purchase the Wellness Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
