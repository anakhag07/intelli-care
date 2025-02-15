import pandas as pd

# Read the CSV file
df = pd.read_csv('heart_attack_vitals.csv')

# Display the dataframe
# print(df)

import streamlit as st
import json
# from flagging_system import check_patient_vitals
# from data_processing import summarize_patient_history

st.title("🚨 Patient Monitoring System")

uploaded_file = st.file_uploader("Upload patient data (JSON)", type="json")
import pandas as pd

# Add CSV uploader option
csv_file = st.file_uploader("Or upload CSV data", type="csv")

if csv_file is not None:
    # Read CSV and convert to JSON format
    df = pd.read_csv(csv_file)
    
    # Convert datetime strings to consistent format
    df['charttime'] = pd.to_datetime(df['charttime']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Convert DataFrame to JSON structure
    patient_data = df.to_dict(orient='records')

    # Check for heart attack symptoms
    heart_attack_detected = False
    for i in range(1, len(patient_data)):
        curr = patient_data[i]
        prev = patient_data[i-1]
        
        # Check for rapid changes in vital signs
        hr_increase = (curr['heartrate'] - prev['heartrate']) > 0
        rr_increase = (curr['resprate'] - prev['resprate']) > 0
        o2_decline = (curr['o2sat'] - prev['o2sat']) < 0
        bp_drop = ((curr['sbp'] - prev['sbp']) < 0) or ((curr['dbp'] - prev['dbp']) < 0)
        
        if hr_increase and rr_increase and o2_decline and bp_drop: #this signals a heart attack warning
            st.markdown("<h1 style='text-align: center; color: red;'>⚠️ HEART ATTACK WARNING ⚠️</h1>", unsafe_allow_html=True)
            heart_attack_detected = True
            break
    
    if not heart_attack_detected:
        st.markdown("<h1 style='text-align: center; color: green;'>🚨 No heart attack symptoms detected 🚨</h1>", unsafe_allow_html=True)
    
    # Allow user to preview the JSON data
    if st.checkbox("Preview JSON data"):
        st.json(patient_data)
    
if uploaded_file is not None:
    patient_data = json.load(uploaded_file)
    # flags = check_patient_vitals(patient_data)
    # summary = summarize_patient_history(patient_data)

    st.subheader("🚨 Alerts")
    # for flag in flags:
    #     st.warning(flag)

    st.subheader("📌 Patient History Summary")
    # st.write(summary)