import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and feature columns
model = pickle.load(open("churn_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
feature_columns = pickle.load(open("feature_columns.pkl","rb"))

# Title
st.title("Customer Churn Prediction")

# Image at top (medium size)
st.image("churn2.jpeg", width=800)

st.write(
"""
This tool predicts whether a customer is **likely to churn or stay**
based on customer activity, purchase behavior, and engagement metrics.
"""
)

st.divider()

st.header("Enter Customer Details")

# Inputs (single column)

Age = st.number_input(
    "Age",
    18, 80,
    help="Customer age in years (18–80)"
)

Login_Frequency = st.number_input(
    "Login Frequency",
    help="Number of times the customer logs into the platform per month (Example: 5–25)"
)

Session_Duration = st.number_input(
    "Session Duration Avg",
    help="Average session duration in minutes (Example: 10–40 minutes)"
)

Pages_Per_Session = st.number_input(
    "Pages Per Session",
    help="Number of pages viewed during one session (Example: 3–12)"
)

Cart_Abandonment_Rate = st.number_input(
    "Cart Abandonment Rate",
    help="Percentage of carts abandoned without purchase (Example: 20–80)"
)

Total_Purchases = st.number_input(
    "Total Purchases",
    help="Total number of purchases made by the customer (Example: 1–50)"
)

Average_Order_Value = st.number_input(
    "Average Order Value",
    help="Average amount spent per order (Example: 50–300)"
)

Days_Since_Last_Purchase = st.number_input(
    "Days Since Last Purchase",
    help="Number of days since last purchase (Example: 5–120)"
)

Mobile_App_Usage = st.number_input(
    "Mobile App Usage",
    help="Weekly usage of the mobile app in minutes (Example: 5–50)"
)

Social_Media_Engagement_Score = st.number_input(
    "Social Media Engagement Score",
    help="Customer engagement score with brand social media (0–100)"
)

st.divider()

# Prediction
if st.button("Predict Churn"):

    input_data = pd.DataFrame(columns=feature_columns)
    input_data.loc[0] = 0

    input_data["Age"] = Age
    input_data["Login_Frequency"] = Login_Frequency
    input_data["Session_Duration_Avg"] = Session_Duration
    input_data["Pages_Per_Session"] = Pages_Per_Session
    input_data["Cart_Abandonment_Rate"] = Cart_Abandonment_Rate
    input_data["Total_Purchases"] = Total_Purchases
    input_data["Average_Order_Value"] = Average_Order_Value
    input_data["Days_Since_Last_Purchase"] = Days_Since_Last_Purchase
    input_data["Mobile_App_Usage"] = Mobile_App_Usage
    input_data["Social_Media_Engagement_Score"] = Social_Media_Engagement_Score

    # Scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(f"Customer likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer likely to stay (Probability: {probability:.2f})")