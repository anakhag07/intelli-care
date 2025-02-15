import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta

def process_vitals(patient_data):
    # Create placeholders for vital signs display
    vitals_display = st.empty()
    warning_display = st.empty()
    chat_display = st.sidebar.empty()
    
    # Add a progress bar
    progress_bar = st.progress(0)
    
    # Initialize last warning time
    last_warning_time = None
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Doctor A: I'm monitoring your patient's vitals. I'll notify you of any concerning changes."}
        ]
    
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
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(patient_data))
        
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
                
                # Check if we should send a new warning message (1-minute intervals)
                if last_warning_time is None or (current_time - last_warning_time) > timedelta(minutes=1):
                    warning_message = {
                        "role": "assistant",
                        "content": f"Doctor A: ⚠️ URGENT: Patient showing signs of cardiac distress!\nHR: {curr['heartrate']}, RR: {curr['resprate']}, O2: {curr['o2sat']}, BP: {curr['sbp']}/{curr['dbp']}"
                    }
                    st.session_state.messages.append(warning_message)
                    last_warning_time = current_time
            else:
                warning_display.empty()
        
        # Display chat messages
        chat_display.markdown("### 💬 Chat with Doctor A")
        for message in st.session_state.messages:
            chat_display.markdown(f"{message['content']}\n")
        
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