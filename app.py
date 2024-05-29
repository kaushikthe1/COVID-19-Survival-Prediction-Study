import joblib
import pandas as pd
import streamlit as st

# Load the saved model, scaler, encoder, and top 5 features
svm_model_top5 = joblib.load('svm_model_top5.pkl')
scaler = joblib.load('scaler.pkl')
top_5_features = joblib.load('top_5_features.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')

# Set the page config
st.set_page_config(page_title="COVID-19 Survival Prediction", page_icon="🏥", layout="centered")

# Set background color to white and adjust the overall style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #CAF4FF;
    }
    .title {
        color: blue;
        font-size: 2.5em;
        font-weight: bold;
        text-align: justify;
        text-justify: inter-word;        
    }
    .description {
        color: gray;
        font-size: 1.2em;
        font-weight: bold;
    }
    .input-label {
        color: black;
        font-size: 1.2em;
        font-weight: bold;
    }
    .input-text, .output-text {
        color: Darkgreen;
        font-size: 1.8em;
        font-weight: bold;
    }
    .stNumberInput > div > div > input {
        width: 200px !important;
        color: black !important;
        background-color: #fff !important;
        border: 2px solid #ff6f61 !important;
        border-radius: 4px !important;
    }
    .stButton > button {
        background-color: #ff6f61 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 4px !important;
        font-size: 1.2em !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown("<div class='title'>COVID-19 Survival Prediction</div>", unsafe_allow_html=True)

# Description of the app
st.markdown(
    """
    <div class='description'>
        A combination of inflammatory and hematological markers strongly predicts mortality in COVID-19 patients, as detailed by Parul Chopra et al.(PMID: 37124653) in their study. Using the same datset, we have developed a SVM model, with accuracy of predicting survival: 80.1%, Sensitivity of 83.7%, and a specificity of 78.6%. These findings highlight the significance of laboratory markers in early prognostication and patient management.
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <div class='description'>
        Please enter the following information to predict the probability of survival in a COVID-19 patient:        
    </div>
    """,
    unsafe_allow_html=True
)


# Input fields
st.markdown("<div class='input-label'>Enter LDH level:</div>", unsafe_allow_html=True)
LDH = st.number_input("", min_value=0.0, step=0.1, format="%f", key="LDH")
st.markdown("<div class='input-label'>Enter IL6 level:</div>", unsafe_allow_html=True)
IL6 = st.number_input("", min_value=0.0, step=0.1, format="%f", key="IL6")
st.markdown("<div class='input-label'>Enter Neutrophil percentage:</div>", unsafe_allow_html=True)
Neutrophil_percentage = st.number_input("", min_value=0.0, step=0.1, format="%f", key="Neutrophil")
st.markdown("<div class='input-label'>Enter Lymphocyte to CRP ratio (LCR):</div>", unsafe_allow_html=True)
LCR = st.number_input("", min_value=0.0, step=0.1, format="%f", key="LCR")
st.markdown("<div class='input-label'>Use the slider to enter severity (severe = 1, not severe = 0):</div>", unsafe_allow_html=True)
severe_1 = st.slider("", 0, 1, key="severe_1")

# Predict button
if st.button("Predict"):
    # Create a DataFrame for the new patient
    new_patient_data = {
        'LDH': [LDH],
        'IL6': [IL6],
        'Neutrophil %': [Neutrophil_percentage],
        'LCR': [LCR],
    }

    # Include missing features with default values (e.g., 0 or median)
    for col in numerical_cols:
        if col not in new_patient_data:
            new_patient_data[col] = [0]  # or any other default value

    new_patient_df = pd.DataFrame(new_patient_data)

    # Standardize numerical columns
    new_patient_df[numerical_cols] = scaler.transform(new_patient_df[numerical_cols])

    # Categorical column
    new_patient_severe = pd.DataFrame({'severe_1': [severe_1]})
    new_patient_df = pd.concat([new_patient_df, new_patient_severe], axis=1)

    # Select the top 5 features
    new_patient_df_top5 = new_patient_df[top_5_features]

    # Make a prediction using the loaded model
    prediction = svm_model_top5.predict(new_patient_df_top5)
    prediction_proba = svm_model_top5.predict_proba(new_patient_df_top5)


    # Display the probability of each class
    st.markdown(f"<div class='output-text'>Probability of Survival (Percent): {round(prediction_proba[0][0]*100, 1)}</div>", unsafe_allow_html=True)
