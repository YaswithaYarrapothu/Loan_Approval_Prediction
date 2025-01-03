import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = 'C:/Users/yaswi/Downloads/loan_approval_dataset.csv'
data = pd.read_csv(data_path)

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Drop unnecessary columns
data = data.drop(columns=['loan_id', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value'])

# Encode categorical features
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_loan_status = LabelEncoder()

data['education'] = le_education.fit_transform(data['education'])
data['self_employed'] = le_self_employed.fit_transform(data['self_employed'])
data['loan_status'] = le_loan_status.fit_transform(data['loan_status'])

# Split dataset into features and target variable
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Streamlit app
st.title("Loan Approval Prediction")

# Display model accuracy
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

st.write("### Enter Applicant Details Below:")

# Define mappings for user-friendly labels to encoded values
education_mapping = {'Educated': 1, 'Not Educated': 0}  # Assuming 'Educated' is 1 and 'Not Educated' is 0
self_employed_mapping = {'Yes': 1, 'No': 0}  # Assuming 'Yes' is 1 and 'No' is 0

# Get user inputs
def user_inputs():
    inputs = {}
    inputs['education'] = st.selectbox('Education', ['Educated', 'Not Educated'])
    inputs['self_employed'] = st.selectbox('Self Employed', ['Yes', 'No'])
    inputs['no_of_dependents'] = st.number_input('Number of Dependents', min_value=0)
    inputs['income_annum'] = st.number_input('Annual Income', min_value=0)
    inputs['loan_amount'] = st.number_input('Loan Amount', min_value=0)
    inputs['loan_term'] = st.number_input('Loan Term (in months)', min_value=0)
    inputs['cibil_score'] = st.number_input('CIBIL Score', min_value=300)
    inputs['bank_asset_value'] = st.number_input('Bank Asset Value', min_value=0)
    
    # Map user inputs to encoded values using mappings
    inputs['education'] = education_mapping[inputs['education']]
    inputs['self_employed'] = self_employed_mapping[inputs['self_employed']]

    # Convert inputs to DataFrame with correct column order
    input_df = pd.DataFrame([inputs], columns=X.columns)
    
    return input_df

user_data = user_inputs()

# Predict based on user input
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.write(f"### Loan Status: {result}")
