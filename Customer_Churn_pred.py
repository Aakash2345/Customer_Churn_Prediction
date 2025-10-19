import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
import matplotlib.pyplot as plt
from xgboost import plot_importance

model = jb.load(r'E://Models//Customer_churn.pkl')

st.title("📊 Customer Churn Prediction App")

st.header("Enter Customer Details")

tab0, tab1 = st.tabs(["Predict Churn", "Feature Importance"])

with tab0:
    with st.expander("Enter Categorical Features"):
        st.subheader("Categorical Features")
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        contract_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        # Map to integer
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        contract_int = contract_map[contract_input]
    
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.subheader("Numerical Features")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)
    num_services = st.number_input("Number of Services Used", min_value=0, max_value=10, value=4)
    TotalSpend = st.number_input("Total Spend ($)", min_value=0.0, max_value=20000.0, value=1800.0)
    AvgSpendPerMonth = st.number_input("Average Spend per Month ($)", min_value=0.0, max_value=1000.0, value=70.0)


# Prepare input dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [contract_int],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'num_services': [num_services],
        'TotalSpend': [TotalSpend],
        'AvgSpendPerMonth': [AvgSpendPerMonth]
    })

    threshold = 0.38

    if st.button("Predict Churn"):
        y_proba = model.predict_proba(input_data)[:, 1][0]
        y_pred_custom = int(y_proba >= threshold)

        if y_pred_custom == 1:
            st.write(f"⚠️ Customer likely to churn")
        else:
            st.write(f"✅ Customer likely to stay")

with tab1:
    # Get the preprocessor from pipeline
    preprocessor = model.named_steps['preprocess']

    # OneHotEncoder for categorical columns
    ohe = preprocessor.named_transformers_['cat']

    # Original categorical columns used in OHE
    categorical_cols = ohe.feature_names_in_

    # Get all OHE column names (after fit)
    ohe_columns = ohe.get_feature_names_out(categorical_cols)

    # Numeric columns (passthrough in your pipeline)
    numeric_cols = ['SeniorCitizen','tenure','Contract','MonthlyCharges','TotalCharges',
                    'num_services','TotalSpend','AvgSpendPerMonth']

    # Full feature names in order
    all_features = list(ohe_columns) + numeric_cols

    # Assign to XGB booster
    xgb_model = model.named_steps['model']
    xgb_model.get_booster().feature_names = all_features
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_importance(xgb_model, max_num_features = 20, importance_type='weight', ax=ax)
    st.pyplot(fig)

        