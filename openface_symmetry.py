import pandas as pd
import numpy as np

# Load OpenFace CSV file
csv_path = "~/Desktop/no_stroke_output/no_stroke.csv"  # Change this to your file path
df = pd.read_csv(csv_path)

# Extract key facial action units related to stroke detection
au_relevant = ["AU12_c", "AU15_c", "AU17_c", "AU20_c", "AU28_c"]
au_data = df[au_relevant]

# Compute overall AU activation score (higher may indicate stroke-like symptoms)
asymmetry_score = np.mean(au_data.values)

# Extract left and right facial landmarks
landmark_cols = [col for col in df.columns if col.startswith("X_") or col.startswith("Y_")]
landmarks = df[landmark_cols].values.reshape(-1, 2, 34)  # Assuming 68 landmarks

# Compute left vs right asymmetry by mirroring half the face
left_side = landmarks[:, :, :17]  # First 17 points (left face)
right_side = landmarks[:, :, 17:]  # Last 17 points (right face, flipped)

# Calculate Euclidean distance between mirrored points
asymmetry_diffs = np.linalg.norm(left_side - np.flip(right_side, axis=2), axis=1)
asymmetry_index = np.mean(asymmetry_diffs)  # Average asymmetry

# Print results
print(f"Facial Asymmetry Index: {asymmetry_index:.2f}")
print(f"Action Unit (AU) Score: {asymmetry_score:.2f}")

# Stroke detection decision
if asymmetry_index > 75 and asymmetry_score > 0.5:
    print("⚠️ Possible stroke detected: High facial asymmetry and muscle drooping.")
else:
    print("✅ No strong stroke indicators detected.")
