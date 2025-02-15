import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from openai_script import generate_vitals_summary

def process_vitals(patient_data):
    # Create placeholders for vital signs display
    vitals_display = st.empty()
    warning_display = st.empty()
    chat_display = st.sidebar.empty()
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
    # Initialize last warning time
    last_warning_time = None
    
    for i in range(len(patient_data)):
        curr = patient_data[i]
        
        # Display current vitals in a formatted way
        vitals_display.markdown(f"""
        ### Current Vital Signs
        - Heart Rate: {curr['heartrate']} bpm
        - Respiratory Rate: {curr['resprate']} breaths/min
        - O2 Saturation: {curr['o2sat']}%
        - Blood Pressure: {curr['sbp']}/{curr['dbp']} mmHg
        """)
        
        # Check for crisis conditions if we have previous data
        if i > 0:
            prev = patient_data[i-1]
            
            # Check for rapid changes in vital signs
            hr_increase = (curr['heartrate'] - prev['heartrate']) > 0
            rr_increase = (curr['resprate'] - prev['resprate']) > 0
            o2_decline = (curr['o2sat'] - prev['o2sat']) < 0
            bp_drop = ((curr['sbp'] - prev['sbp']) < 0) or ((curr['dbp'] - prev['dbp']) < 0)
            
            current_time = datetime.now()
            
            if hr_increase and rr_increase and o2_decline and bp_drop:
                warning_display.markdown("<h1 style='text-align: center; color: red;'>⚠️ HEART ATTACK WARNING ⚠️</h1>", unsafe_allow_html=True)
                
                # Generate summary using OpenA
                critical_vitals = {
                    "timestamp": curr['charttime'],
                    "vitals": {
                        "temperature": curr['temperature'],
                        "heart_rate": curr['heartrate'],
                        "respiratory_rate": curr['resprate'], 
                        "oxygen_saturation": curr['o2sat'],
                        "blood_pressure": {
                            "systolic": curr['sbp'],
                            "diastolic": curr['dbp']
                        },
                        "heart_rhythm": curr['rhythm']
                    },
                    "changes_detected": {
                        "heart_rate_increase": hr_increase,
                        "respiratory_rate_increase": rr_increase,
                        "oxygen_decline": o2_decline,
                        "blood_pressure_drop": bp_drop
                    }
                }
                summary = generate_vitals_summary("", critical_vitals) #MAKE SURE YOU PUT IN UR OPENAI API KEY
                
                # Create JSON with current vitals
                
                warning_message = {
                    "role": "assistant",
                    "content": f"⚠️ URGENT: Heart Attack Warning Detected!\n\nPatient Summary:\n{summary}"
                }
                chat_display.markdown(f"{warning_message['content']}\n")
                
                # Stop monitoring after detecting heart attack
                break
            else:
                warning_display.empty()
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(patient_data))
        
        # Add a small delay to simulate real-time monitoring
        time.sleep(1)

def main():
    st.title("Patient Vital Signs Monitor")
    
    uploaded_file = st.file_uploader("Upload patient vitals CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        patient_data = df.to_dict('records')
        
        # Add a start button
        if st.button("Start Monitoring"):
            process_vitals(patient_data)

if __name__ == "__main__":
    main()