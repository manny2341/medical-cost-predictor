# 💊 Medical Cost Predictor

A regression machine learning web app that predicts annual healthcare insurance costs based on patient demographics and health factors. Emphasises feature importance analysis to explain what actually drives medical costs.

## Results

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | ~75% | ~$4,200 | ~$6,100 |
| Ridge Regression | ~75% | ~$4,200 | ~$6,100 |
| Lasso Regression | ~74% | ~$4,300 | ~$6,200 |
| Random Forest | ~86% | ~$2,600 | ~$4,700 |
| **Gradient Boosting ⭐** | **~88%** | **~$2,400** | **~$4,500** |

## Key Finding: Smoking Drives Cost

| Group | Average Annual Cost |
|-------|-------------------|
| Non-Smokers | ~$8,400 |
| Smokers | ~$32,000 |
| **Difference** | **~$23,600 more for smokers** |

Smoking alone accounts for over 60% of prediction variance — the single strongest driver of medical costs in this dataset.

## Features

- 5 regression models compared: Linear, Ridge, Lasso, Random Forest, Gradient Boosting
- Feature importance analysis showing what drives healthcare costs
- Key insight banner: cost difference between smokers and non-smokers
- R², MAE, RMSE evaluation metrics
- Interactive prediction form with dropdowns
- Automatic dataset download on first run

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Scikit-Learn |
| Preprocessing | StandardScaler, LabelEncoder |
| Evaluation | R², MAE, RMSE |
| Web Framework | Flask |
| Dataset | Medical Cost Personal Dataset (1,338 records) |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/medical-cost-predictor.git
cd medical-cost-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the app**
```bash
python3 app.py
```
The dataset downloads automatically on first run.

**4. Open in browser**
```
http://127.0.0.1:5010
```

## Dataset Features

| Feature | Description |
|---------|-------------|
| age | Patient age |
| sex | male / female |
| bmi | Body Mass Index |
| children | Number of dependants |
| smoker | yes / no |
| region | northeast / northwest / southeast / southwest |
| charges | Annual medical insurance cost (target) |

Dataset: [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) — 1,338 records.

## Project Structure

```
medical-cost-predictor/
├── app.py               # Flask server, regression pipeline, prediction API
├── templates/
│   └── index.html       # Model comparison, feature importance, prediction form
├── static/
│   └── style.css        # Dark medical theme
└── requirements.txt
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Diabetes Classifier | Model comparison + feature scaling demo | [diabetes-classifier](https://github.com/manny2341/diabetes-classifier) |
| Heart Attack Predictor | End-to-end classification pipeline | [heart-attack-predictor](https://github.com/manny2341/heart-attack-predictor) |
| Stock Price Predictor | LSTM forecasting for 5,884 tickers + crypto | [stock-price-predictor](https://github.com/manny2341/stock-price-predictor) |
| Crop Disease Detector | EfficientNetV2 — 15 plant diseases | [crop-disease-detector](https://github.com/manny2341/crop-disease-detector) |

## Author

[@manny2341](https://github.com/manny2341)
