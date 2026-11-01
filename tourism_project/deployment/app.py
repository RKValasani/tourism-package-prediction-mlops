import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_package_model_v1.joblib"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The trained model is not present. Run the GitHub Actions "
            "training workflow and ensure it commits the model file."
        )
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️")
st.title("Wellness Tourism Package Prediction")
st.write(
    "Enter the customer and interaction details to estimate whether the "
    "customer is likely to purchase the tourism package."
)

# Input categories are taken from the supplied tourism.csv dataset.
with st.form("prediction_form"):
    st.subheader("Customer and interaction details")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=61, value=37)
        TypeofContact = st.selectbox(
            "Type of Contact", ["Self Enquiry", "Company Invited"]
        )
        CityTier = st.selectbox("City Tier", [1, 2, 3], index=0)
        Occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Free Lancer", "Small Business", "Large Business"]
        )
        Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])
        NumberOfPersonVisiting = st.number_input(
            "Number of Persons Visiting", min_value=1, max_value=5, value=3
        )
        PreferredPropertyStar = st.selectbox(
            "Preferred Property Star", [3, 4, 5], index=0
        )
        MaritalStatus = st.selectbox(
            "Marital Status", ["Single", "Divorced", "Married", "Unmarried"]
        )
        NumberOfTrips = st.number_input(
            "Number of Trips", min_value=1, max_value=22, value=3
        )
    with col2:
        Passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        OwnCar = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        NumberOfChildrenVisiting = st.number_input(
            "Number of Children Visiting", min_value=0, max_value=3, value=1
        )
        Designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        MonthlyIncome = st.number_input(
            "Monthly Income", min_value=1000, max_value=98678, value=23000, step=500
        )
        PitchSatisfactionScore = st.selectbox(
            "Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2
        )
        ProductPitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
        )
        NumberOfFollowups = st.number_input(
            "Number of Followups", min_value=1, max_value=6, value=4
        )
        DurationOfPitch = st.number_input(
            "Duration of Pitch", min_value=5, max_value=127, value=15
        )

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "ProductPitched": ProductPitched,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch
    }])

    model = load_model()
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0, 1])

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The model predicts that the customer is likely to purchase the package.")
    else:
        st.info("The model predicts that the customer is unlikely to purchase the package.")
    st.metric("Estimated purchase probability", f"{probability:.1%}")
