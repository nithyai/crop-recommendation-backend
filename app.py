from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import numpy as np

app = FastAPI(title="Smart Crop Recommendation API")
@app.get("/")
def home():
    return {"message": "Backend is running"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler and models
scaler = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_scaler.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_rf_model.pkl"))
cat_model = joblib.load(os.path.join(BASE_DIR, "CatBoost_model.joblib"))

# CatBoost label mapping
cat_label_mapping = {
    0: "rice",
    1: "wheat",
    2: "maize",
    3: "cotton",
    4: "jute",
    5: "groundnut",
    6: "apple"
}

# Crop nutrient requirements
crop_requirements = {
    "rice": {"N": 80, "P": 40, "K": 40},
    "wheat": {"N": 70, "P": 35, "K": 40},
    "maize": {"N": 60, "P": 30, "K": 30},
    "cotton": {"N": 65, "P": 35, "K": 35}
}

# Input model
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Deficiency detection
def detect_deficiency(crop, N, P, K):
    deficiencies = []
    req = crop_requirements.get(crop.lower())
    if req:
        if N < req["N"]:
            deficiencies.append("Nitrogen deficiency")
        if P < req["P"]:
            deficiencies.append("Phosphorus deficiency")
        if K < req["K"]:
            deficiencies.append("Potassium deficiency")
    return deficiencies

# Fertilizer recommendation
def recommend_fertilizer(deficiencies):
    fertilizers = []
    for d in deficiencies:
        if "Nitrogen" in d:
            fertilizers.append("Apply Urea")
        if "Phosphorus" in d:
            fertilizers.append("Apply DAP")
        if "Potassium" in d:
            fertilizers.append("Apply MOP")
    return fertilizers

# Weather-based care
def weather_care(temperature, rainfall):
    care = []
    if temperature > 35:
        care.append("Increase irrigation and provide shade")
    if rainfall < 50:
        care.append("Use drip irrigation and mulching")
    if rainfall > 300:
        care.append("Improve field drainage")
    return care

# Dynamic crop care
def dynamic_crop_care(crop, N, P, K, temperature, rainfall):
    deficiencies = detect_deficiency(crop, N, P, K)
    fertilizers = recommend_fertilizer(deficiencies)
    care = weather_care(temperature, rainfall)
    return deficiencies, fertilizers, care

@app.post("/predict")
def predict_crop(data: CropInput):
    try:
        # Normalize integers to 0-1 if model was trained on normalized data
        N_scaled = data.N / 140
        P_scaled = data.P / 140
        K_scaled = data.K / 200
        temp_scaled = data.temperature / 50
        hum_scaled = data.humidity / 100
        ph_scaled = data.ph / 14
        rain_scaled = data.rainfall / 300

        features = np.array([[N_scaled, P_scaled, K_scaled, temp_scaled, hum_scaled, ph_scaled, rain_scaled]])
        X_scaled = scaler.transform(features).astype(float)

        # Random Forest top 3 crops
        rf_probs = rf_model.predict_proba(X_scaled)[0]  # array of probabilities
        rf_top_indices = np.argsort(rf_probs)[::-1][:3]  # top 3 indices
        rf_top_crops = [
            {"crop": rf_model.classes_[i], "confidence": round(rf_probs[i]*100, 2)}
            for i in rf_top_indices
        ]

        # CatBoost top 3 crops
        cat_probs = cat_model.predict_proba(X_scaled)[0]
        cat_top_indices = np.argsort(cat_probs)[::-1][:3]
        cat_top_crops = [
            {"crop": cat_label_mapping[i], "confidence": round(cat_probs[i]*100, 2)}
            for i in cat_top_indices
        ]


        # Use top-1 CatBoost crop for crop care logic
        primary_crop = cat_top_crops[0]["crop"]

        deficiencies, fertilizers, care = dynamic_crop_care(
            primary_crop,
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.rainfall
        )

        return {
        "random_forest": rf_top_crops,
        "catboost": cat_top_crops,
        "nutrient_deficiencies": deficiencies,
        "fertilizer_recommendations": fertilizers,
        "weather_based_care": care
        }



    except Exception as e:
        return {
            "error": str(e),
            "hint": "Check scaler, model compatibility, and feature order"
        }
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
