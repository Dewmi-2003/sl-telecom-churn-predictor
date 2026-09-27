import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    feature_names = joblib.load('feature_names.pkl')
    explainer = shap.TreeExplainer(model)
    return model, preprocessor, feature_names, explainer

model, preprocessor, feature_names, explainer = load_artifacts()

st.set_page_config(page_title="SL Telecom Churn Advisor", layout="wide")
st.title("🇱🇰 Sri Lanka Telecom & Broadband Churn Predictor")
st.write("Predict subscriber cancellation risk and view key risk drivers powered by Explainable AI (SHAP).")

# Sidebar Controls
st.sidebar.header("Subscriber Details")
tenure = st.sidebar.slider("Account Tenure (Months)", 1, 72, 12)
monthly_bill = st.sidebar.slider("Monthly Bill (LKR)", 1500, 15000, 4500, step=500)
support_tickets = st.sidebar.slider("Support / Network Complaints (Last 30 Days)", 0, 10, 3)

connection_type = st.sidebar.selectbox("Connection Type", ['Fibre', '4G Home Broadband', 'Postpaid Mobile'])
payment_method = st.sidebar.selectbox("Payment Channel", ['eZ Cash / Frimi / e-Wallet', 'Bank Direct Debit', 'Over the Counter / Cash'])

input_df = pd.DataFrame([{
    'Tenure_Months': tenure,
    'Monthly_Bill_LKR': monthly_bill,
    'Support_Tickets_30d': support_tickets,
    'Connection_Type': connection_type,
    'Payment_Method': payment_method
}])

processed_input = preprocessor.transform(input_df)
processed_df = pd.DataFrame(processed_input, columns=feature_names)

churn_prob = model.predict_proba(processed_df)[0][1]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Subscriber Churn Risk")
    st.metric(label="Churn Probability", value=f"{churn_prob * 100:.1f}%")
    
    if churn_prob > 0.5:
        st.error("🚨 **Status:** High Churn Risk")
        st.write("**Retention Action:** Trigger 20% Bill Discount or offer Free Speed Upgrade for 3 Months.")
    else:
        st.success("✅ **Status:** Low Risk Account")
        st.write("**Retention Action:** Standard Loyalty Campaign.")

with col2:
    st.subheader("Risk Explanation (SHAP Values)")
    shap_values = explainer(processed_df)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)