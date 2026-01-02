import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("End-to-end ML system with monitoring & drift detection")

# ======================
# Input form
# ======================
with st.form("churn_form"):
    CustomerID = st.number_input("Customer ID", value=1001)
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Tenure = st.number_input("Tenure (months)", value=12)
    Usage_Frequency = st.number_input("Usage Frequency", value=10)
    Support_Calls = st.number_input("Support Calls", value=1)
    Payment_Delay = st.number_input("Payment Delay (days)", value=0)
    Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    Contract_Length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
    Total_Spend = st.number_input("Total Spend", value=2000.0)
    Last_Interaction = st.number_input("Last Interaction (days ago)", value=5)

    submitted = st.form_submit_button("Predict Churn")

# ======================
# Prediction
# ======================
if submitted:
    payload = {
        "CustomerID": CustomerID,
        "Age": Age,
        "Gender": Gender,
        "Tenure": Tenure,
        "Usage_Frequency": Usage_Frequency,
        "Support_Calls": Support_Calls,
        "Payment_Delay": Payment_Delay,
        "Subscription_Type": Subscription_Type,
        "Contract_Length": Contract_Length,
        "Total_Spend": Total_Spend,
        "Last_Interaction": Last_Interaction
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        st.subheader("📊 Prediction Result")
        st.write(f"**Churn Probability:** {result['Churn_Probability']}")
        st.write(f"**Prediction:** {result['Meaning']}")

        if result["Drift_Detected"]:
            st.warning("⚠️ Data drift detected!")
            st.write("Drifted features:", result["Drifted_Features"])
        else:
            st.success("✅ No data drift detected")

    except Exception as e:
        st.error("API not reachable. Is FastAPI running?")
