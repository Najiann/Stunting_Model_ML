# Stunting Prediction using Machine Learning
### Improved Model with Data Leakage Mitigation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Laravel](https://img.shields.io/badge/Laravel-12-red)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Overview

This project predicts stunting status in toddlers using Machine Learning based on anthropometric, nutritional, health, and socio-economic features.

The project is an improved version of the baseline model by addressing a critical machine learning issue known as **data leakage**. The final model was retrained without the `risk_score` feature, resulting in a prediction pipeline that relies only on information realistically available during screening.

---

## Related Repository

This repository focuses on the **Machine Learning model** and **FastAPI prediction service**.

The complete Laravel web application that integrates this model is available at:

**Stunting Prediction (Laravel Web Application)**

https://github.com/Najiann/Stunting_Prediction

---

## Disclaimer

This project was developed as part of a personal learning and model improvement process based on the baseline project provided during the Dev Weekend program.

During development, the author utilized **Artificial Intelligence (AI)** tools, including **ChatGPT**, as a **development assistant** to support activities such as:

- brainstorming solutions
- understanding Machine Learning concepts
- debugging
- code review
- implementation suggestions
- documentation writing
- improving code structure and readability

All data analysis, data leakage investigation, feature engineering, model selection, hyperparameter tuning, implementation, testing, evaluation, deployment, and final technical decisions were carried out, reviewed, and validated by the author.

AI was used solely as a development assistant and learning aid, **not as a replacement for the developer**.

---

## Model Improvements

Compared to the baseline model, the following improvements were implemented:

| Aspect | Baseline | Improved Model |
|---------|----------|----------------|
| Data Leakage Analysis | Not investigated | Added formal leakage investigation |
| `risk_score` Feature | Used for training | Removed after leakage analysis |
| Number of Models | 3 algorithms | 6 algorithms |
| Hyperparameter Tuning | Default parameters | GridSearchCV & RandomizedSearchCV |
| Cross Validation | Train-test split only | Stratified 5-Fold Cross Validation |
| Imbalanced Data Handling | `class_weight` only | `class_weight`, `scale_pos_weight`, and SMOTE |
| Model Selection | Limited comparison | Final model selected after all experiments |
| Deployment | Requires `risk_score` | No longer requires `risk_score` |

---

## Project Structure

```text
ML/
├── stunting_ml_analysis_fixed_no_leakage.ipynb
├── main.py
├── stunting_model_fix_no_leakage.pkl
├── dataset_stunting_ml_1000.csv
├── generate_preprocessors.py
├── requirements.txt
└── README.md
```

---

## Project Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Leakage Investigation
   │
   ▼
Feature Selection
(Remove risk_score)
   │
   ▼
Data Preprocessing
(Label Encoding + Standard Scaling)
   │
   ▼
Model Training
(Logistic Regression, Random Forest,
Gradient Boosting, XGBoost,
LightGBM, KNN)
   │
   ▼
Hyperparameter Tuning
(GridSearchCV & RandomizedSearchCV)
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
FastAPI Deployment
   │
   ▼
Laravel Integration
```

---

## Dataset

The dataset contains **1,000 toddler records** consisting of demographic, nutritional, health, and environmental information.

### Features

| Feature | Description |
|----------|-------------|
| usia_bulan | Age (months) |
| jenis_kelamin | Gender |
| berat_lahir_kg | Birth weight |
| panjang_lahir_cm | Birth length |
| asi_eksklusif | Exclusive breastfeeding |
| protein_harian | Daily protein intake |
| frekuensi_makan | Daily meal frequency |
| tinggi_ibu_cm | Mother's height |
| riwayat_diare | History of diarrhea |
| pendapatan_keluarga | Family income |
| sanitasi_layak | Access to proper sanitation |
| imunisasi_lengkap | Complete immunization |
| status_stunting | Target variable |

> The `risk_score` feature was intentionally removed from the final model because it was identified as a potential source of data leakage.

---

## Model Development

The notebook includes:

- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Data Leakage Investigation
- Feature Selection
- Data Preprocessing
- Model Comparison
- Hyperparameter Tuning
- Cross Validation
- SMOTE Experiment
- Final Model Selection
- Model Evaluation

The final model was selected after comparing all experiments using multiple evaluation metrics, with particular attention to Recall and F1-Score for stunting screening.

---

## API

The prediction service is implemented using **FastAPI**.

### Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health Check |
| POST | `/predict` | Predict stunting status |

### Example Request

```json
{
  "usia_bulan": 24,
  "jenis_kelamin": "L",
  "berat_lahir_kg": 3.2,
  "panjang_lahir_cm": 50,
  "asi_eksklusif": "Ya",
  "protein_harian": 45,
  "frekuensi_makan": 4,
  "tinggi_ibu_cm": 160,
  "riwayat_diare": 0,
  "pendapatan_keluarga": 6000000,
  "sanitasi_layak": "Ya",
  "imunisasi_lengkap": "Ya"
}
```

### Example Response

```json
{
  "status": "success",
  "prediction_code": 0,
  "prediction_status": "Tidak Stunting",
  "probability_stunting_percent": 12.45
}
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload --port 8001
```

Swagger Documentation:

```text
http://127.0.0.1:8001/docs
```

---

## Laravel Integration

Example request using Laravel HTTP Client.

```php
$response = Http::timeout(10)->post(
    'http://127.0.0.1:8001/predict',
    [
        'usia_bulan' => 24,
        'jenis_kelamin' => 'L',
        'berat_lahir_kg' => 3.2,
        'panjang_lahir_cm' => 50,
        'asi_eksklusif' => 'Ya',
        'protein_harian' => 45,
        'frekuensi_makan' => 4,
        'tinggi_ibu_cm' => 160,
        'riwayat_diare' => 0,
        'pendapatan_keluarga' => 6000000,
        'sanitasi_layak' => 'Ya',
        'imunisasi_lengkap' => 'Ya'
    ]
);

$result = $response->json();
```

---

## Documentation

A detailed report describing the complete model development process, data leakage investigation, model comparison, and evaluation is available in:

```text
docs/
└── MODEL_IMPROVEMENT_REPORT.pdf
```

---

## Technologies

- Python
- FastAPI
- Scikit-Learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## License

This project was developed for educational purposes and machine learning experimentation.
