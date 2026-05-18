# Global Health & Mortality Prediction System

A professional Flask machine learning web application that predicts **Death Rate** and **Life Expectancy** using Random Forest models trained on global health, lifestyle, healthcare, economic, pollution, and nutrition indicators.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)

## Project Overview

This system enables researchers and public health professionals to:

- Submit regional health and socioeconomic indicators
- Receive ML-powered predictions for mortality and life expectancy
- View health risk levels and personalized recommendations
- Explore project methodology on the About page

## Folder Structure

```
health-prediction-app/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── model/
│   ├── death_rate_model.pkl
│   ├── life_expectancy_model.pkl
│   └── model_columns.pkl
├── data/
│   └── health_dataset.csv
├── templates/
│   ├── index.html
│   ├── result.html
│   └── about.html
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
└── notebook/
    └── health_prediction.ipynb
```

## Technologies Used

| Category | Tools |
|----------|-------|
| Backend | Python, Flask |
| Machine Learning | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn |
| Frontend | HTML5, CSS3, Bootstrap 5, Font Awesome |

## Installation

1. **Clone or navigate to the project directory:**

```bash
cd health-prediction-app
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## How to Run

1. **Train the machine learning models:**

```bash
python train_model.py
```

This generates `data/health_dataset.csv` (if missing) and saves trained models to `model/`.

2. **Start the Flask application:**

```bash
python app.py
```

3. **Open your browser:**

```
http://127.0.0.1:5000
```

## Usage

1. Open the home page and scroll to the **Health Prediction Form**
2. Enter values for all 18 input features (year, gender, lifestyle, healthcare, environment, nutrition)
3. Click **Generate Prediction**
4. View predicted death rate, life expectancy, risk level, and health recommendations on the results page

## Input Features

- Year, Gender, Total Population
- Alcohol Abuse, Tobacco Prevalence
- GDP per Capita, Doctors, Nurses and Midwifes
- Universal Health Care Coverage, Government Expenditure Health, Birth Rate
- Air Pollution Death Rate Total, Basic Drinking Water Services, Clean Fuel and Technology
- Vegetable Consumption, Fruit Consumption Apples, Diet Calories Fat, Diet Calories Carbohydrates

## Machine Learning Models

- **Model 1:** RandomForestRegressor → Death Rate
- **Model 2:** RandomForestRegressor → Life Expectancy

Models use 200 estimators with preprocessing: missing value imputation and one-hot encoding for Gender.

## Screenshots

> Add screenshots of the home page, prediction form, and results page after running the app.

| Page | Description |
|------|-------------|
| Home | Hero section, statistics cards, prediction form |
| Results | Predicted metrics, risk level, recommendations |
| About | Project objectives, models, dataset, technologies |

## License

This project is provided for educational and research purposes.
