import streamlit as st
import pandas as pd
import pickle

# ── Load artifacts ──
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))
threshold = pickle.load(open("best_threshold.pkl", "rb"))

# ── Page config ──
st.set_page_config(page_title="Churn Prediction", layout="centered")

# ── Title ──
st.title("📊 Customer Churn Prediction")

# ── Image ──
st.image("churn2.jpeg", use_container_width=True)

st.markdown("""
Predict whether a customer is **likely to churn or stay**  
based on behavioral and engagement data.
""")

st.divider()

# ── Input Section ──
st.header("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 80)
    Login_Frequency = st.number_input("Login Frequency")
    Session_Duration = st.number_input("Session Duration Avg")
    Pages_Per_Session = st.number_input("Pages Per Session")
    Cart_Abandonment_Rate = st.number_input("Cart Abandonment Rate")

with col2:
    Total_Purchases = st.number_input("Total Purchases")
    Average_Order_Value = st.number_input("Average Order Value")
    Days_Since_Last_Purchase = st.number_input("Days Since Last Purchase")
    Mobile_App_Usage = st.number_input("Mobile App Usage")
    Social_Media_Engagement_Score = st.number_input("Social Media Engagement Score")

st.divider()

# ── Prediction ──
if st.button("🚀 Predict Churn"):

    try:
        # Create input dataframe
        input_data = pd.DataFrame(columns=feature_columns)
        input_data.loc[0] = 0

        # Assign values
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

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict probability
        probability = model.predict_proba(input_scaled)[0][1]

        # Apply tuned threshold
        prediction = int(probability >= threshold)

        st.subheader("📢 Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ Likely to Churn (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Likely to Stay (Probability: {probability:.2f})")

        # Extra info (matches your training output)
        st.caption(f"""
        Model Used: XGBoost (Tuned)  
        ROC-AUC: ~0.93  
        Threshold: {threshold}
        """)

    except Exception as e:
        st.error(f"Error during prediction: {e}")