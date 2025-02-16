import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from openai_script import generate_vitals_summary
import lightgbm as lgb

OPEN_API_KEY = "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"

def process_vitals(patient_data):
    """Process vitals using AI model with dynamic risk scoring"""
    # Load the trained model
    model = lgb.Booster(model_file='best_hrvarrest_model.txt')
    
    # Create placeholders for vital signs display
    vitals_display = st.empty()
    warning_display = st.empty()
    chat_display = st.sidebar.empty()
    risk_display = st.empty()  # New display for risk score
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
    for i in range(len(patient_data)):
        curr = patient_data[i]
        
        # Display current vitals in a formatted way
        vitals_display.markdown(f"""
        ### Current Vital Signs
        - Heart Rate: {curr['heartrate']} bpm
        - Respiratory Rate: {curr['resprate']} breaths/min
        - O2 Saturation: {curr['o2sat']}%
        - Blood Pressure: {curr['sbp']}/{curr['dbp']} mmHg
        - Temperature: {curr['temperature']}°F
        """)
        
        # Calculate changes from previous timestep
        if i > 0:
            prev = patient_data[i-1]
            hr_change = curr['heartrate'] - prev['heartrate']
            rr_change = curr['resprate'] - prev['resprate']
            o2_change = curr['o2sat'] - prev['o2sat']
        else:
            hr_change = 0
            rr_change = 0
            o2_change = 0
        
        # Create feature DataFrame for prediction
        features = pd.DataFrame({
            'temperature': [curr['temperature']],
            'heartrate': [curr['heartrate']],
            'resprate': [curr['resprate']],
            'o2sat': [curr['o2sat']],
            'sbp': [curr['sbp']],
            'dbp': [curr['dbp']],
            'pain': [curr['pain']],
            'hour': [-1],
            'day_of_week': [-1],
            'hr_change': [hr_change],
            'rr_change': [rr_change],
            'o2_change': [o2_change]
        })
        
        # Get model prediction for current timestep
        risk_score = float(model.predict(features))
        
        # Display current risk score with color coding
        risk_color = "green" if risk_score < 0.3 else "orange" if risk_score < 0.7 else "red"
        risk_display.markdown(f"""
        ### Current Risk Assessment
        <p style='color: {risk_color}; font-size: 20px;'>
            Risk Score: {risk_score:.3f}
        </p>
        """, unsafe_allow_html=True)
        
        # Check if risk score exceeds threshold
        print(risk_score)
        if risk_score > 0.5:  # High risk threshold
            warning_display.markdown("<h1 style='text-align: center; color: red;'>⚠️ CARDIAC EVENT WARNING ⚠️</h1>", unsafe_allow_html=True)
            
            # Calculate changes for the summary
            changes_detected = {
                "heart_rate": f"Current: {curr['heartrate']} bpm",
                "respiratory_rate": f"Current: {curr['resprate']} breaths/min",
                "oxygen_saturation": f"Current: {curr['o2sat']}%",
                "blood_pressure": f"Current: {curr['sbp']}/{curr['dbp']} mmHg"
            }
            
            if i > 0:
                changes_detected.update({
                    "heart_rate": f"Changed by {hr_change} bpm",
                    "respiratory_rate": f"Changed by {rr_change} breaths/min",
                    "oxygen_saturation": f"Changed by {o2_change}%",
                    "blood_pressure": f"Systolic changed by {curr['sbp'] - prev['sbp']}, Diastolic changed by {curr['dbp'] - prev['dbp']}"
                })
            
            # Convert timestamp to proper format
            timestamp = pd.to_datetime(curr['charttime']).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(curr.get('charttime')) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            critical_vitals = {
                "timestamp": timestamp,
                "vitals": {
                    "temperature": curr['temperature'],
                    "heart_rate": curr['heartrate'],
                    "respiratory_rate": curr['resprate'],
                    "oxygen_saturation": curr['o2sat'],
                    "blood_pressure": {
                        "systolic": curr['sbp'],
                        "diastolic": curr['dbp']
                    },
                    "heart_rhythm": curr.get('rhythm', 'Unknown')
                },
                "risk_score": risk_score,
                "model_prediction": "High Risk of Cardiac Event",
                "changes_detected": changes_detected
            }

            summary = generate_vitals_summary(OPEN_API_KEY, critical_vitals)
            warning_message = {
                "role": "assistant",
                "content": f"⚠️ URGENT: Cardiac Event Warning!\n\nRisk Score: {risk_score:.4f}\n\nPatient Summary:\n{summary}"
            }
            chat_display.markdown(f"{warning_message['content']}\n")
            break
        else:
            warning_display.empty()
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(patient_data))
        
        # Add a small delay to simulate real-time monitoring
        time.sleep(1)

def process_vitals_rule_based(patient_data):
    """Process vitals using rule-based monitoring"""
    # Create placeholders for vital signs display
    vitals_display = st.empty()
    warning_display = st.empty()
    chat_display = st.sidebar.empty()
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
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
        
        if i > 0:
            prev = patient_data[i-1]
            
            # Check for rapid changes in vital signs
            hr_increase = (curr['heartrate'] - prev['heartrate']) > 0
            rr_increase = (curr['resprate'] - prev['resprate']) > 0
            o2_decline = (curr['o2sat'] - prev['o2sat']) < 0
            bp_drop = ((curr['sbp'] - prev['sbp']) < 0) or ((curr['dbp'] - prev['dbp']) < 0)
            
            if hr_increase and rr_increase and o2_decline and bp_drop:
                warning_display.markdown("<h1 style='text-align: center; color: red;'>⚠️ HEART ATTACK WARNING ⚠️</h1>", unsafe_allow_html=True)
                
                # Generate summary using OpenAI
                # Convert timestamp to proper format
                timestamp = pd.to_datetime(curr['charttime']).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(curr.get('charttime')) else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                critical_vitals = {
                    "timestamp": timestamp,
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
                    "risk_score": 1.0,  # High risk for rule-based
                    "model_prediction": "High Risk of Cardiac Event (Rule-based)"
                }
                
                summary = generate_vitals_summary(OPEN_API_KEY, critical_vitals)
                warning_message = {
                    "role": "assistant",
                    "content": f"⚠️ URGENT: Cardiac Event Warning!\n\nRisk Assessment: Rule-based Detection\n\nPatient Summary:\n{summary}"
                }
                chat_display.markdown(f"{warning_message['content']}\n")
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
        
        # Add monitoring method selection
        monitoring_method = st.radio(
            "Select Monitoring Method",
            ["AI Model", "Rule-based"],
            help="Choose between AI model prediction or rule-based monitoring"
        )
        
        # Add a start button
        if st.button("Start Monitoring"):
            if monitoring_method == "AI Model":
                process_vitals(patient_data)  # Original AI-based monitoring
            else:
                process_vitals_rule_based(patient_data)  # New rule-based monitoring

if __name__ == "__main__":
    main()