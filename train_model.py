"""
Train Random Forest models for Death Rate and Life Expectancy prediction.
Generates sample dataset if health_dataset.csv is missing.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "health_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURE_COLUMNS = [
    "Year",
    "Gender",
    "Alcohol Abuse",
    "Tobacco Prevalence",
    "GDP per Capita",
    "Doctors",
    "Nurses and Midwifes",
    "Air Pollution Death Rate Total",
    "Universal Heath Care Coverage",
    "Birth Rate",
    "Government Expenditure Health",
    "Vegetable Consumption",
    "Fruit Consumption Apples",
    "Diet Calories Fat",
    "Diet Calories Carbohydrates",
    "Basic Drinking Water Services",
    "Clean Fuel and Technology",
    "Total Population",
]

TARGET_DEATH = "Death Rate"
TARGET_LIFE = "Life Expectancy"


def generate_sample_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic global health dataset for training."""
    rng = np.random.default_rng(seed)
    genders = ["Male", "Female", "Both"]

    years = rng.integers(2010, 2024, size=n_samples)
    gender = rng.choice(genders, size=n_samples, p=[0.35, 0.35, 0.30])
    alcohol = rng.uniform(2, 18, size=n_samples)
    tobacco = rng.uniform(5, 45, size=n_samples)
    gdp = rng.uniform(800, 75000, size=n_samples)
    doctors = rng.uniform(0.2, 8, size=n_samples)
    nurses = rng.uniform(1, 15, size=n_samples)
    air_pollution = rng.uniform(5, 120, size=n_samples)
    uhc = rng.uniform(20, 100, size=n_samples)
    birth_rate = rng.uniform(8, 45, size=n_samples)
    gov_health = rng.uniform(2, 12, size=n_samples)
    vegetables = rng.uniform(50, 350, size=n_samples)
    fruit = rng.uniform(20, 180, size=n_samples)
    fat_cal = rng.uniform(25, 45, size=n_samples)
    carb_cal = rng.uniform(40, 65, size=n_samples)
    water = rng.uniform(55, 100, size=n_samples)
    clean_fuel = rng.uniform(30, 100, size=n_samples)
    population = rng.uniform(100_000, 80_000_000, size=n_samples)

    # Synthetic targets with realistic relationships
    death_rate = (
        4.5
        + tobacco * 0.12
        + alcohol * 0.08
        + air_pollution * 0.04
        - doctors * 0.35
        - nurses * 0.15
        - uhc * 0.03
        - water * 0.02
        - clean_fuel * 0.015
        + (80000 - np.clip(gdp, 800, 80000)) / 12000
        - gov_health * 0.2
        + rng.normal(0, 1.2, size=n_samples)
    )
    death_rate = np.clip(death_rate, 2, 25)

    # Life expectancy correlates with inputs and inversely with mortality
    life_expectancy = (
        55
        + doctors * 2.5
        + nurses * 1.0
        + uhc * 0.15
        + water * 0.10
        + clean_fuel * 0.06
        + vegetables * 0.018
        + fruit * 0.012
        + gov_health * 1.2
        + np.log1p(gdp) * 2.0
        - tobacco * 0.20
        - alcohol * 0.12
        - air_pollution * 0.05
        - death_rate * 0.85
        + rng.normal(0, 0.8, size=n_samples)
    )
    life_expectancy = np.clip(life_expectancy, 48, 88)

    df = pd.DataFrame(
        {
            "Year": years,
            "Gender": gender,
            "Alcohol Abuse": np.round(alcohol, 2),
            "Tobacco Prevalence": np.round(tobacco, 2),
            "GDP per Capita": np.round(gdp, 2),
            "Doctors": np.round(doctors, 2),
            "Nurses and Midwifes": np.round(nurses, 2),
            "Air Pollution Death Rate Total": np.round(air_pollution, 2),
            "Universal Heath Care Coverage": np.round(uhc, 2),
            "Birth Rate": np.round(birth_rate, 2),
            "Government Expenditure Health": np.round(gov_health, 2),
            "Vegetable Consumption": np.round(vegetables, 2),
            "Fruit Consumption Apples": np.round(fruit, 2),
            "Diet Calories Fat": np.round(fat_cal, 2),
            "Diet Calories Carbohydrates": np.round(carb_cal, 2),
            "Basic Drinking Water Services": np.round(water, 2),
            "Clean Fuel and Technology": np.round(clean_fuel, 2),
            "Total Population": np.round(population, 0),
            TARGET_DEATH: np.round(death_rate, 2),
            TARGET_LIFE: np.round(life_expectancy, 2),
        }
    )
    return df


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset, handle missing values, and encode categoricals."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print("Dataset not found. Generating sample health_dataset.csv...")
        df = generate_sample_dataset()
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Handle missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # One-hot encode Gender (keeps Male, Female, Both columns)
    df = pd.get_dummies(df, columns=["Gender"], drop_first=False)

    numeric_features = [c for c in FEATURE_COLUMNS if c != "Gender"]
    gender_cols = sorted([c for c in df.columns if c.startswith("Gender_")])
    feature_cols = numeric_features + gender_cols

    X = df[feature_cols]
    y_death = df[TARGET_DEATH]
    y_life = df[TARGET_LIFE]

    return df, X, y_death, y_life


def train_and_save_models() -> None:
    """Train two RandomForestRegressor models and persist artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    _, X, y_death, y_life = load_and_prepare_data()
    model_columns = list(X.columns)

    X_train, X_test, yd_train, yd_test, yl_train, yl_test = train_test_split(
        X, y_death, y_life, test_size=0.2, random_state=42
    )

    death_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    life_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    death_model.fit(X_train, yd_train)
    life_model.fit(X_train, yl_train)

    # Evaluation
    death_pred = death_model.predict(X_test)
    life_pred = life_model.predict(X_test)

    print("\n=== Death Rate Model ===")
    print(f"MAE: {mean_absolute_error(yd_test, death_pred):.3f}")
    print(f"R2:  {r2_score(yd_test, death_pred):.3f}")

    print("\n=== Life Expectancy Model ===")
    print(f"MAE: {mean_absolute_error(yl_test, life_pred):.3f}")
    print(f"R2:  {r2_score(yl_test, life_pred):.3f}")

    with open(os.path.join(MODEL_DIR, "death_rate_model.pkl"), "wb") as f:
        pickle.dump(death_model, f)
    with open(os.path.join(MODEL_DIR, "life_expectancy_model.pkl"), "wb") as f:
        pickle.dump(life_model, f)
    with open(os.path.join(MODEL_DIR, "model_columns.pkl"), "wb") as f:
        pickle.dump(model_columns, f)

    print(f"\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    train_and_save_models()
