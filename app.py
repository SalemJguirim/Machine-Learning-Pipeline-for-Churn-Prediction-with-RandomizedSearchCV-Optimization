import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load('churn_model.pkl')  # Your model path
label_encoders = joblib.load('label_encoders.pkl')  # Your encoders path

# Page Config
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# Light theme styling
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    .main {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 6px;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>📞 Telco Customer Churn Prediction</h1><br>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.header("📂 Navigation")
menu = st.sidebar.radio(
    "Choose a section:",
    ["📊 Upload Dataset", "📈 Visualize Data", "🔮 Predict Churn", "📝 About"],
    index=0,
)

# 📊 Upload Dataset
if menu == "📊 Upload Dataset":
    st.subheader("📤 Upload a CSV File")
    uploaded_file = st.file_uploader("Upload your Telco dataset (.csv)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.success("✅ File uploaded successfully!")
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

# 📈 Visualize Data
elif menu == "📈 Visualize Data":
    st.subheader("📊 Data Visualization")
    if 'data' in st.session_state:
        data = st.session_state['data']
        feature = st.selectbox("Choose a feature to visualize:", data.columns)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload a dataset first.")

# 🔮 Predict Churn
elif menu == "🔮 Predict Churn":
    st.subheader("🔍 Predict Churn for a Customer")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            Partner = st.selectbox("Partner", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

        with col2:
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)

        submitted = st.form_submit_button("🔮 Predict")

        if submitted:
            input_data = pd.DataFrame([{
                'gender': gender,
                'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
            }])

            try:
                for col in input_data.columns:
                    if col in label_encoders and input_data[col].dtype == 'object':
                        input_data[col] = label_encoders[col].transform(input_data[col])
                prediction = model.predict(input_data)[0]
                result = "✅ Yes (Customer Will Churn)" if prediction == 1 else "❌ No (Customer Will Stay)"
                st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{result}</h3>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# 📝 About
elif menu == "📝 About":
    st.subheader("📘 About This App")
    st.write("""
    This web app uses a machine learning model to predict whether a customer will churn based on their telco service details.

    **Main Features:**
    - Upload and explore Telco datasets
    - Visualize any feature
    - Predict churn for a single customer
    
    **Developed By:** Salem Jguirim
    """)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f9f9f9;
        padding: 8px;
        text-align: center;
        font-size: 13px;
        color: #888;
    }
    </style>
    <div class="footer">
        Developed by <b>Salem Jguirim</b> | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
