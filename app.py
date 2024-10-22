import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
RESULTS_PATH = "results.pkl"
ENCODERS_PATH = "encoders.pkl"


def load_data():
    path = "dataset/insurance.csv"
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()
    encoders = {}
    for col in ["sex", "smoker", "region"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_models():
    df = load_data()
    df_enc, encoders = preprocess(df)

    feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
    X = df_enc[feature_cols]
    y = df_enc["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    best_r2 = -999
    best_model = None

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        r2 = round(r2_score(y_test, y_pred) * 100, 2)
        mae = round(mean_absolute_error(y_test, y_pred), 2)
        rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 2)
        results.append({"model": name, "r2": r2, "mae": mae, "rmse": rmse})
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Feature importance from Gradient Boosting
    gb = models["Gradient Boosting"]
    feat_imp = sorted(zip(feature_cols, gb.feature_importances_), key=lambda x: x[1], reverse=True)

    # Key insights
    smoker_mean = round(df[df["smoker"] == "yes"]["charges"].mean(), 2)
    non_smoker_mean = round(df[df["smoker"] == "no"]["charges"].mean(), 2)
    avg_cost = round(df["charges"].mean(), 2)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    all_results = {
        "model_results": results,
        "feature_importance": feat_imp,
        "dataset_size": len(df),
        "avg_cost": avg_cost,
        "smoker_mean": smoker_mean,
        "non_smoker_mean": non_smoker_mean,
        "best_model": max(results, key=lambda x: x["r2"])["model"]
    }
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)

    print("Training complete.")
    for r in results:
        print(f"  {r['model']}: R²={r['r2']}% MAE=${r['mae']}")
    return all_results


if os.path.exists(MODEL_PATH) and os.path.exists(RESULTS_PATH):
    print("Loading cached model...")
    with open(RESULTS_PATH, "rb") as f:
        RESULTS = pickle.load(f)
else:
    print("Training models...")
    RESULTS = train_models()

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    ENCODERS = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html", results=RESULTS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        age = float(data["age"])
        sex = 1 if data["sex"] == "male" else 0
        bmi = float(data["bmi"])
        children = int(data["children"])
        smoker = 1 if data["smoker"] == "yes" else 0
        region_map = {"northeast": 2, "northwest": 3, "southeast": 4, "southwest": 5}
        region = region_map.get(data["region"], 2)

        X = np.array([[age, sex, bmi, children, smoker, region]])
        X_sc = SCALER.transform(X)
        pred = float(MODEL.predict(X_sc)[0])

        return jsonify({
            "predicted_cost": round(pred, 2),
            "formatted": f"${pred:,.2f}",
            "risk_level": "High" if pred > 20000 else "Medium" if pred > 10000 else "Low",
            "smoker_impact": smoker == 1
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5010)
