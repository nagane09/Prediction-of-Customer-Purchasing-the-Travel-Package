import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

# -----------------------------
# Load Saved Model + Preprocessor
# -----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    st.error("Model or preprocessor files are missing. Make sure model.joblib and preprocessor.joblib are in the app folder.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    st.error(f"Failed to load model or preprocessor: {e}")
    st.stop()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Travel Package Prediction", layout="wide")

st.markdown("""
    <h2 style='text-align:center; color:#2E86C1;'>Travel Package Purchase Prediction</h2>
    <p style='text-align:center;'>Fill in the customer information in each section to get a prediction.</p>
""", unsafe_allow_html=True)

# -----------------------------
# 1️⃣ Personal Details
# -----------------------------
with st.expander("🧍 Personal Details", expanded=False):
    col1, col2, col3 = st.columns(3)
    Age = col1.number_input("Age", min_value=18, max_value=90, value=30)
    Gender = col2.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = col3.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])

# -----------------------------
# 2️⃣ Travel Interest
# -----------------------------
with st.expander("🌍 Travel Interest", expanded=False):
    col1, col2, col3 = st.columns(3)
    ProductPitched = col1.selectbox("Product Pitched", ["Basic", "Deluxe"])
    PreferredPropertyStar = col2.selectbox("Preferred Property Star", [1,2,3,4,5])
    NumberOfTrips = col3.number_input("Number of Trips", min_value=0, value=1)

# -----------------------------
# 3️⃣ Pitch Information
# -----------------------------
with st.expander("📞 Pitch Information", expanded=False):
    col1, col2, col3 = st.columns(3)
    TypeofContact = col1.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    DurationOfPitch = col2.number_input("Duration of Pitch (minutes)", min_value=0, value=5)
    NumberOfFollowups = col3.number_input("Number of Followups", min_value=0, value=2)

    col4, col5, col6 = st.columns(3)
    PitchSatisfactionScore = col4.selectbox("Pitch Satisfaction Score", [1,2,3,4,5])
    Designation = col5.selectbox("Designation", ["Executive", "Manager"])
    CityTier = col6.selectbox("City Tier", [1,2,3])

# -----------------------------
# 4️⃣ Financial & Other Details
# -----------------------------
with st.expander("💰 Financial & Other Details", expanded=False):
    col1, col2, col3 = st.columns(3)
    Occupation = col1.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business"])
    Passport = col2.selectbox("Passport", [0,1])
    OwnCar = col3.selectbox("Own Car", [0,1])

    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=20000)
    TotalVisiting = st.number_input("Total Visiting", min_value=0, value=2)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict", use_container_width=True):

    # Prepare input
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "TotalVisiting": TotalVisiting
    }])

    # Prediction
    try:
        transformed = preprocessor.transform(input_data)
        prediction = model.predict(transformed)[0]
        st.success("Prediction Completed!")

        if prediction == 1:
            st.markdown("<h3 style='color: green;'>✔ Customer is LIKELY to purchase the travel package.</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>✘ Customer is NOT likely to purchase the travel package.</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
