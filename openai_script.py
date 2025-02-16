import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

from datetime import datetime

OPEN_API_KEY = "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"

# Define structured response format
class PatientSummary(BaseModel):
    patient_id: str
    # date: str
    # time: str
    temperature: float
    heart_rate: float
    respiratory_rate: float
    oxygen_saturation: float
    sys_blood_pressure: str
    dia_blood_pressure: str
    heart_rhythm: str
    past_history: list[str]
    summary: str  # Summary of the patient's vitals and history
def generate_vitals_summary(api_key, patient_data):
    client = OpenAI(api_key=api_key)
    # Read the CSV file
    # df = pd.read_csv(patient_data_file)
    # Convert patient data to DataFrame format
    df = pd.DataFrame([patient_data])
    
    # Extract data from the first row since we're processing a single record
    row = df.iloc[0]
    
    # Parse timestamp
    # timestamp = row['timestamp']
    # dt_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    # date_part = dt_obj.strftime("%Y-%m-%d") 
    # time_part = dt_obj.strftime("%H:%M")
    
    # Extract vitals from nested dictionary
    vitals = row['vitals']
    # changes = row['changes_detected']
    
    past_history = ["heart attack"]

    patient_info = f"""
    Patient ID: {row['patient_id'] if 'patient_id' in row else 'Unknown'}
    Temperature: {vitals['temperature']} °F
    Heart Rate: {vitals['heart_rate']} bpm
    Respiratory Rate: {vitals['respiratory_rate']} breaths/min
    Oxygen Saturation: {vitals['oxygen_saturation']}%
    Systolic Blood Pressure: {vitals['blood_pressure']['systolic']}
    Diastolic Blood Pressure: {vitals['blood_pressure']['diastolic']}
    Heart Rhythm: {vitals['heart_rhythm']}
    Past Medical History: {past_history}
    """
    # Call OpenAI API to summarize patient vitals
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a medical assistant. Generate a concise summary of the patient's vital signs and health status."},
            {"role": "user", "content": patient_info},
        ],
        response_format=PatientSummary,
    )

        # Extract structured response
    patient_summary = completion.choices[0].message.parsed
    return patient_summary.summary

