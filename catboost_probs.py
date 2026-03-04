import pandas as pd
import joblib  # for loading the model and scaler
import numpy as np

# ----------------------------
# 1. Load the trained CatBoost model
# ----------------------------
catboost_model = joblib.load("CatBoost_model.joblib")

# If you used a scaler during training, load it
scaler = joblib.load("crop_recommendation_scaler.pkl")

# ----------------------------
# 2. Load your dataset or user input
# ----------------------------
# Example: use first 5 rows of your CSV
df = pd.read_csv("Crop_pre.csv")

# Features used for training
feature_cols = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']

# Scale features exactly like training
X_scaled = scaler.transform(df[feature_cols])

# ----------------------------
# 3. Predict probabilities for all crops
# ----------------------------
probs = catboost_model.predict_proba(X_scaled)

# Convert to percentages
probs_percent = probs * 100

# Get the crop names from the model
crop_classes = catboost_model.classes_

# Create a DataFrame to see probabilities nicely
probs_df = pd.DataFrame(probs_percent, columns=crop_classes)

# ----------------------------
# 4. Show predictions for first 5 samples
# ----------------------------
print("Predicted probabilities (in %) for first 5 samples:")
print(probs_df.head().round(2))  # round to 2 decimal places

# ----------------------------
# 5. Show predicted crop (highest probability)
# ----------------------------
predicted_crops = probs_df.idxmax(axis=1)
print("\nPredicted crops for first 5 samples:")
print(predicted_crops.head())
