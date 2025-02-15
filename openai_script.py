import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

from datetime import datetime

# Initialize OpenAI client
client = OpenAI()

# Define structured response format
class PatientSummary(BaseModel):
    patient_id: str
    date: str
    time: str
    temperature: float
    heart_rate: float
    respiratory_rate: float
    oxygen_saturation: float
    sys_blood_pressure: str
    dia_blood_pressure: str
    heart_rhythm: str
    past_history: list[str]
    summary: str  # Summary of the patient's vitals and history

# Read the CSV file
df = pd.read_csv("patient_a_heart_attack_vitals.csv")

# Iterate over each patient record
for _, row in df.iterrows():
    # Construct input message for GPT
    timestamp = row['charttime']
    dt_obj = datetime.strptime(timestamp, "%m/%d/%y %H:%M")
    date_part = dt_obj.strftime("%Y-%m-%d")  # Output: '2025-02-15'
    time_part = dt_obj.strftime("%H:%M")     # Output: '08:00'
    past_history = ["heart attack"]

    patient_info = f"""
    Patient ID: {row['subject_id']}
    Date: {date_part}
    Time: {time_part}
    Temperature: {row['temperature']} °F
    Heart Rate: {row['heartrate']} bpm
    Respiratory Rate: {row['resprate']} breaths/min
    Oxygen Saturation: {row['o2sat']}%
    Systolic Blood Pressure: {row['sbp']}
    Diastolic Blood Pressure: {row['dbp']}
    Heart Rhythm: {row['rhythm']}
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
    print(f"Summary for {row['subject_id']}:\n{patient_summary.summary}\n")

