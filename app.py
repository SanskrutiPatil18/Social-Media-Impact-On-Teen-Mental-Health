import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
rf_model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

# Define the order of columns the model expects
# This should match the order of X_scaled from training
# This is crucial for correct prediction with one-hot encoding
expected_columns = ['age', 'daily_social_media_hours', 'sleep_hours',
                    'screen_time_before_sleep', 'academic_performance',
                    'physical_activity', 'stress_level', 'anxiety_level',
                    'addiction_level', 'gender_male', 'platform_usage_Instagram',
                    'platform_usage_TikTok', 'social_interaction_level_low',
                    'social_interaction_level_medium']

st.title('Teen Mental Health: Depression Prediction')
st.write('Enter the teen\'s information to predict the likelihood of depression.')

# --- Input Features --- 

# Age input with a slider
age = st.slider('Age', min_value=12, max_value=20, value=15)

# Gender selection
gender = st.selectbox('Gender', ['female', 'male'])

# Daily Social Media Hours input
daily_social_media_hours = st.slider('Daily Social Media Hours', min_value=0.0, max_value=10.0, value=3.0, step=0.1)

# Platform Usage selection
platform_usage = st.selectbox('Primary Platform Usage', ['Instagram', 'TikTok', 'Both'])

# Sleep Hours input
sleep_hours = st.slider('Sleep Hours', min_value=4.0, max_value=12.0, value=7.0, step=0.1)

# Screen Time Before Sleep input
screen_time_before_sleep = st.slider('Screen Time Before Sleep (hours)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# Academic Performance input
academic_performance = st.slider('Academic Performance (GPA equivalent)', min_value=0.0, max_value=4.0, value=2.5, step=0.1)

# Physical Activity input
physical_activity = st.slider('Physical Activity (hours/day)', min_value=0.0, max_value=3.0, value=0.5, step=0.1)

# Social Interaction Level selection
social_interaction_level = st.selectbox('Social Interaction Level', ['low', 'medium', 'high'])

# Stress Level input
stress_level = st.slider('Stress Level (1-10)', min_value=1, max_value=10, value=5)

# Anxiety Level input
anxiety_level = st.slider('Anxiety Level (1-10)', min_value=1, max_value=10, value=5)

# Addiction Level input
addiction_level = st.slider('Addiction Level (1-10)', min_value=1, max_value=10, value=5)

# --- Preprocessing Function --- 
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    df_input = pd.DataFrame([input_data])

    # One-hot encode categorical features
    # Make sure to handle all possible categories and potential missing ones during prediction
    # Recreate the columns that were dropped_first
    df_input['gender_male'] = 1 if df_input['gender'].iloc[0] == 'male' else 0
    df_input['platform_usage_Instagram'] = 1 if df_input['platform_usage'].iloc[0] == 'Instagram' else 0
    df_input['platform_usage_TikTok'] = 1 if df_input['platform_usage'].iloc[0] == 'TikTok' else 0
    
    # For 'social_interaction_level', we need to create two columns as 'high' is the reference
    df_input['social_interaction_level_low'] = 1 if df_input['social_interaction_level'].iloc[0] == 'low' else 0
    df_input['social_interaction_level_medium'] = 1 if df_input['social_interaction_level'].iloc[0] == 'medium' else 0

    # Drop original categorical columns
    df_input = df_input.drop(columns=['gender', 'platform_usage', 'social_interaction_level'])

    # Select and reorder columns to match the training data
    # This is critical. Ensure `expected_columns` is correct
    final_input = df_input[expected_columns].copy()

    # Apply scaling to numerical columns
    numerical_cols = ['age', 'daily_social_media_hours', 'sleep_hours',
                      'screen_time_before_sleep', 'academic_performance',
                      'physical_activity', 'stress_level', 'anxiety_level',
                      'addiction_level']

    final_input[numerical_cols] = scaler.transform(final_input[numerical_cols])

    return final_input

# --- Prediction --- 
if st.button('Predict'):
    input_data = {
        'age': age,
        'gender': gender,
        'daily_social_media_hours': daily_social_media_hours,
        'platform_usage': platform_usage,
        'sleep_hours': sleep_hours,
        'screen_time_before_sleep': screen_time_before_sleep,
        'academic_performance': academic_performance,
        'physical_activity': physical_activity,
        'social_interaction_level': social_interaction_level,
        'stress_level': stress_level,
        'anxiety_level': anxiety_level,
        'addiction_level': addiction_level,
    }

    processed_input = preprocess_input(input_data)
    prediction = rf_model.predict(processed_input)
    prediction_proba = rf_model.predict_proba(processed_input)[:, 1]

    st.write('---')
    if prediction[0] == 1:
        st.error(f"Prediction: High likelihood of Depression (Probability: {prediction_proba[0]:.2f})")
        st.write("It is recommended to seek professional help or support.")
    else:
        st.success(f"Prediction: Low likelihood of Depression (Probability: {prediction_proba[0]:.2f})")
        st.write("Continue to monitor mental well-being.")

    st.write('---')
    st.write("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice or diagnosis.")
