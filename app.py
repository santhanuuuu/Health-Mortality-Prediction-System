"""
Global Health & Mortality Prediction System
Flask web application for predicting death rate and life expectancy.
Maps simple health-survey inputs to ML model features internally.
"""

import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np # type: ignore
import pandas as pd # type: ignore
from flask import Flask, flash, redirect, render_template, request, url_for # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "global-health-mortality-prediction-2026")

death_model = None
life_model = None
model_columns: list[str] = []

# Allowed simple survey values
SURVEY_CHOICES = {
    "gender": {"Male", "Female"},
    "smoking_habit": {"non_smoker", "occasionally", "regular_smoker"},
    "alcohol_consumption": {"never", "sometimes", "frequently"},
    "exercise_level": {"low", "moderate", "high"},
    "water_intake": {"low", "medium", "high"},
    "fast_food": {"rarely", "weekly", "daily"},
    "stress_level": {"low", "medium", "high"},
    "pollution_exposure": {"low", "medium", "high"},
    "healthcare_access": {"poor", "average", "good"},
    "healthy_food": {"poor", "average", "good"},
}


def load_models() -> None:
    """Load trained ML models and feature column order."""
    global death_model, life_model, model_columns

    death_path = os.path.join(MODEL_DIR, "death_rate_model.pkl")
    life_path = os.path.join(MODEL_DIR, "life_expectancy_model.pkl")
    cols_path = os.path.join(MODEL_DIR, "model_columns.pkl")

    if not all(os.path.exists(p) for p in [death_path, life_path, cols_path]):
        raise FileNotFoundError(
            "Trained models not found. Run: python train_model.py"
        )

    with open(death_path, "rb") as f:
        death_model = pickle.load(f)
    with open(life_path, "rb") as f:
        life_model = pickle.load(f)
    with open(cols_path, "rb") as f:
        model_columns = pickle.load(f)


def validate_form(data: dict[str, str]) -> list[str]:
    """Validate simple health survey inputs."""
    errors: list[str] = []

    age_raw = data.get("age", "").strip()
    if not age_raw:
        errors.append("Age is required.")
    else:
        try:
            age = int(float(age_raw))
            if age < 1 or age > 120:
                errors.append("Age must be between 1 and 120.")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number.")

    sleep_raw = data.get("sleep_hours", "").strip()
    if not sleep_raw:
        errors.append("Sleep hours is required.")
    else:
        try:
            sleep = float(sleep_raw)
            if sleep < 3 or sleep > 12:
                errors.append("Sleep hours must be between 3 and 12.")
        except (ValueError, TypeError):
            errors.append("Sleep hours must be a valid number.")

    for field, allowed in SURVEY_CHOICES.items():
        value = data.get(field, "").strip()
        if not value:
            label = field.replace("_", " ").title()
            errors.append(f"{label} is required.")
        elif value not in allowed:
            errors.append(f"Please select a valid option for {field.replace('_', ' ')}.")

    return errors


def map_simple_inputs_to_features(form: dict[str, str]) -> dict[str, float]:
    """
    Map beginner-friendly survey answers to approximate dataset feature values
    used by the trained Random Forest models.
    """
    age = int(float(form["age"]))
    gender = form.get("gender", "Female")
    smoking = form.get("smoking_habit", "non_smoker")
    alcohol = form.get("alcohol_consumption", "sometimes")
    exercise = form.get("exercise_level", "moderate")
    water = form.get("water_intake", "medium")
    sleep = float(form.get("sleep_hours", 7))
    fast_food = form.get("fast_food", "rarely")
    stress = form.get("stress_level", "medium")
    pollution = form.get("pollution_exposure", "medium")
    healthcare = form.get("healthcare_access", "average")
    healthy_food = form.get("healthy_food", "average")

    # Smoking → Tobacco Prevalence
    tobacco_map = {"non_smoker": 5.0, "occasionally": 15.0, "regular_smoker": 38.0}
    tobacco = tobacco_map[smoking]

    # Alcohol → Alcohol Abuse
    alcohol_map = {"never": 2.0, "sometimes": 7.0, "frequently": 16.0}
    alcohol_abuse = alcohol_map[alcohol]

    # Healthy food → vegetables & fruit
    food_map = {
        "poor": (85.0, 40.0),
        "average": (165.0, 85.0),
        "good": (245.0, 125.0),
    }
    vegetables, fruit = food_map[healthy_food]

    # Fast food → diet composition
    fast_food_map = {
        "rarely": (28.0, 50.0),
        "weekly": (35.0, 56.0),
        "daily": (42.0, 62.0),
    }
    diet_fat, diet_carbs = fast_food_map[fast_food]

    # Pollution exposure → air pollution death rate
    pollution_map = {"low": 18.0, "medium": 42.0, "high": 78.0}
    air_pollution = pollution_map[pollution]

    # Healthcare access → UHC, doctors, nurses, GDP
    healthcare_map = {
        "poor": {"uhc": 42.0, "doctors": 1.2, "nurses": 2.5, "gdp": 6000.0, "gov": 3.5},
        "average": {"uhc": 72.0, "doctors": 2.6, "nurses": 5.2, "gdp": 15000.0, "gov": 5.5},
        "good": {"uhc": 92.0, "doctors": 4.2, "nurses": 8.0, "gdp": 38000.0, "gov": 7.5},
    }
    hc = healthcare_map[healthcare]

    # Water intake → drinking water services
    water_map = {"low": 68.0, "medium": 86.0, "high": 96.0}
    water_services = water_map[water]

    # Exercise → workforce & clean fuel proxy
    exercise_map = {
        "low": {"clean_fuel": 58.0, "doctor_boost": -0.4, "nurse_boost": -0.8},
        "moderate": {"clean_fuel": 72.0, "doctor_boost": 0.0, "nurse_boost": 0.0},
        "high": {"clean_fuel": 88.0, "doctor_boost": 0.5, "nurse_boost": 1.0},
    }
    ex = exercise_map[exercise]
    doctors = hc["doctors"] + ex["doctor_boost"]
    nurses = hc["nurses"] + ex["nurse_boost"]
    clean_fuel = ex["clean_fuel"]

    # Sleep & stress adjustments
    gov_health = hc["gov"]
    if sleep < 6:
        gov_health -= 0.8
    elif sleep >= 8:
        gov_health += 0.4
    if stress == "high":
        gov_health -= 0.6
    elif stress == "low":
        gov_health += 0.3

    # Age → birth rate proxy (population demographic signal)
    if age < 25:
        birth_rate = 22.0
    elif age < 40:
        birth_rate = 17.0
    elif age < 55:
        birth_rate = 13.0
    else:
        birth_rate = 9.0

    return {
        "Year": float(datetime.now().year),
        "Alcohol Abuse": alcohol_abuse,
        "Tobacco Prevalence": tobacco,
        "GDP per Capita": hc["gdp"],
        "Doctors": max(0.5, doctors),
        "Nurses and Midwifes": max(1.0, nurses),
        "Air Pollution Death Rate Total": air_pollution,
        "Universal Heath Care Coverage": hc["uhc"],
        "Birth Rate": birth_rate,
        "Government Expenditure Health": max(2.0, gov_health),
        "Vegetable Consumption": vegetables,
        "Fruit Consumption Apples": fruit,
        "Diet Calories Fat": diet_fat,
        "Diet Calories Carbohydrates": diet_carbs,
        "Basic Drinking Water Services": water_services,
        "Clean Fuel and Technology": clean_fuel,
        "Total Population": 1_000_000.0,
        "Gender_Male": 1.0 if gender == "Male" else 0.0,
        "Gender_Female": 1.0 if gender == "Female" else 0.0,
        "Gender_Both": 0.0,
    }


def build_feature_dataframe(form: dict[str, str]) -> pd.DataFrame:
    """Convert survey form into ML-ready feature DataFrame."""
    mapped = map_simple_inputs_to_features(form)
    df = pd.DataFrame([mapped])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df[model_columns]


def _label(value: str) -> str:
    """Turn form value into readable label."""
    return value.replace("_", " ").title()


def build_survey_summary(form: dict[str, str]) -> list[dict[str, str]]:
    """Human-readable summary of survey answers for results page."""
    return [
        {"label": "Age", "value": form.get("age", "")},
        {"label": "Gender", "value": form.get("gender", "")},
        {"label": "Smoking", "value": _label(form.get("smoking_habit", ""))},
        {"label": "Alcohol", "value": _label(form.get("alcohol_consumption", ""))},
        {"label": "Exercise", "value": _label(form.get("exercise_level", ""))},
        {"label": "Water Intake", "value": _label(form.get("water_intake", ""))},
        {"label": "Sleep", "value": f"{form.get('sleep_hours', '')} hours/night"},
        {"label": "Fast Food", "value": _label(form.get("fast_food", ""))},
        {"label": "Stress", "value": _label(form.get("stress_level", ""))},
        {"label": "Pollution Exposure", "value": _label(form.get("pollution_exposure", ""))},
        {"label": "Healthcare Access", "value": _label(form.get("healthcare_access", ""))},
        {"label": "Healthy Food", "value": _label(form.get("healthy_food", ""))},
    ]


def assess_health_risk(
    death_rate: float, life_expectancy: float, form: dict[str, str]
) -> dict[str, Any]:
    """Determine health risk level and personalized recommendations in plain language."""
    if death_rate >= 12:
        risk_level = "High Health Risk"
        risk_message = "Your lifestyle indicates a high health risk."
        risk_class = "risk-high"
        risk_icon = "fa-triangle-exclamation"
    elif death_rate >= 8:
        risk_level = "Moderate Health Risk"
        risk_message = "Your lifestyle indicates a moderate health risk."
        risk_class = "risk-moderate"
        risk_icon = "fa-circle-exclamation"
    else:
        risk_level = "Low Health Risk"
        risk_message = "Your lifestyle indicates a low health risk. Keep up the good work!"
        risk_class = "risk-low"
        risk_icon = "fa-shield-heart"

    if life_expectancy >= 75:
        health_status = "Excellent Health Outlook"
        status_message = "Your habits support a strong, healthy life expectancy."
        status_class = "status-healthy"
    elif life_expectancy >= 65:
        health_status = "Good Health Outlook"
        status_message = "You're on a positive track — small improvements can go a long way."
        status_class = "status-average"
    else:
        health_status = "Health Needs Attention"
        status_message = "Your assessment suggests room to improve daily wellness habits."
        status_class = "status-needs-improvement"

    recommendations: list[str] = []

    if form.get("smoking_habit") == "regular_smoker":
        recommendations.append(
            "Try to reduce smoking — even cutting back can significantly improve your long-term health."
        )
    elif form.get("smoking_habit") == "occasionally":
        recommendations.append(
            "Consider avoiding occasional smoking to lower your health risk over time."
        )

    if form.get("alcohol_consumption") == "frequently":
        recommendations.append(
            "Limit alcohol to recommended levels and take alcohol-free days each week."
        )

    if form.get("exercise_level") == "low":
        recommendations.append(
            "Aim for at least 150 minutes of moderate activity per week, such as brisk walking."
        )

    if form.get("healthy_food") == "poor":
        recommendations.append(
            "Add more vegetables, fruits, and whole grains to your daily meals."
        )

    if form.get("fast_food") in ("weekly", "daily"):
        recommendations.append(
            "Reduce fast food and choose home-cooked meals with fresh ingredients more often."
        )

    if form.get("water_intake") == "low":
        recommendations.append(
            "Drink more water throughout the day — aim for 6–8 glasses daily."
        )

    sleep = float(form.get("sleep_hours", 7))
    if sleep < 6:
        recommendations.append(
            "Prioritize 7–8 hours of sleep each night for better recovery and mental clarity."
        )

    if form.get("stress_level") == "high":
        recommendations.append(
            "Practice stress relief through meditation, hobbies, or talking with someone you trust."
        )

    if form.get("pollution_exposure") == "high":
        recommendations.append(
            "Limit time in polluted areas and use masks or air filters when exposure is unavoidable."
        )

    if form.get("healthcare_access") == "poor":
        recommendations.append(
            "Schedule regular check-ups when possible and learn about local community health programs."
        )

    if not recommendations:
        recommendations.append(
            "Maintain your healthy habits and revisit this assessment periodically to track progress."
        )

    return {
        "risk_level": risk_level,
        "risk_message": risk_message,
        "risk_class": risk_class,
        "risk_icon": risk_icon,
        "health_status": health_status,
        "status_message": status_message,
        "status_class": status_class,
        "recommendations": recommendations,
    }


def _calibrate_predictions(pred_death: float, pred_life: float, features_df: pd.DataFrame) -> tuple[float, float]:
    """Calibrate model outputs using a lightweight risk score so results
    better reflect extreme (very good / very poor) input combinations.

    This is a small post-processing step (not a model re-train) that
    increases separation for clearly low/high risk feature sets.
    """
    try:
        f = features_df.iloc[0]
        tobacco = float(f.get("Tobacco Prevalence", 10.0))
        alcohol = float(f.get("Alcohol Abuse", 5.0))
        air = float(f.get("Air Pollution Death Rate Total", 30.0))
        doctors = float(f.get("Doctors", 2.5))
        nurses = float(f.get("Nurses and Midwifes", 5.0))
        uhc = float(f.get("Universal Heath Care Coverage", 70.0))
        water = float(f.get("Basic Drinking Water Services", 80.0))

        # normalize to 0-1 using reasonable bounds from training data
        tobacco_n = np.clip((tobacco - 5.0) / 40.0, 0.0, 1.0)
        alcohol_n = np.clip((alcohol - 2.0) / 20.0, 0.0, 1.0)
        air_n = np.clip((air - 5.0) / 120.0, 0.0, 1.0)
        doctors_n = 1.0 - np.clip((doctors - 0.2) / 8.0, 0.0, 1.0)
        nurses_n = 1.0 - np.clip((nurses - 1.0) / 15.0, 0.0, 1.0)
        uhc_n = 1.0 - np.clip((uhc - 20.0) / 80.0, 0.0, 1.0)
        water_n = 1.0 - np.clip((water - 55.0) / 45.0, 0.0, 1.0)

        # weighted risk score (0 low risk -> 1 high risk)
        risk_score = (
            0.30 * tobacco_n
            + 0.20 * alcohol_n
            + 0.20 * air_n
            + 0.10 * doctors_n
            + 0.10 * nurses_n
            + 0.05 * uhc_n
            + 0.05 * water_n
        )

        # baseline expected risk roughly in the middle; compute adjustment
        adj = risk_score - 0.20

        # Apply soft calibration: boost death for high risk, reduce life expectancy
        pred_death = float(pred_death) * (1.0 + 0.35 * adj)
        pred_life = float(pred_life) * (1.0 - 0.8 * adj)

        return pred_death, pred_life
    except Exception:
        return pred_death, pred_life


@app.before_request
def ensure_models_loaded():
    """Load models on first request if not already loaded."""
    global death_model, life_model, model_columns
    if death_model is None:
        load_models()


@app.route("/")
def index():
    """Home page with hero, statistics, and prediction form."""
    stats = {
        "countries_analyzed": "190+",
        "health_indicators": "18",
        "ml_models": "2",
        "accuracy_note": "R² > 0.85",
    }
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    """Process simple survey form, map to ML features, and render results."""
    errors = validate_form(request.form)
    if errors:
        for msg in errors:
            flash(msg, "danger")
        return redirect(url_for("index") + "#prediction-form")

    try:
        features = build_feature_dataframe(request.form)
        pred_death = float(death_model.predict(features)[0])
        pred_life = float(life_model.predict(features)[0])

        # Calibrate model outputs with a lightweight risk score to increase
        # separation for clearly low/high risk inputs (post-processing only).
        pred_death, pred_life = _calibrate_predictions(pred_death, pred_life, features)

        pred_death = round(max(0, pred_death), 2)
        pred_life = round(np.clip(pred_life, 40, 95), 1)

        health = assess_health_risk(pred_death, pred_life, request.form)

        context = {
            "death_rate": pred_death,
            "life_expectancy": pred_life,
            "survey_summary": build_survey_summary(request.form),
            **health,
        }
        return render_template("result.html", **context)
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "danger")
        return redirect(url_for("index"))


@app.route("/about")
def about():
    """About page with project and technology information."""
    return render_template("about.html")


if __name__ == "__main__":
    try:
        load_models()
        print("Models loaded successfully.")
    except FileNotFoundError as err:
        print(err)
        print("Training models now...")
        from train_model import train_and_save_models

        train_and_save_models()
        load_models()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)