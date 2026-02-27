# 🏠 AI House Price Predictor — FAANG-Style Regression System

An end-to-end machine learning web application that predicts house prices using advanced feature engineering, automated preprocessing, and a premium interactive Streamlit UI.

Built to demonstrate **production-grade ML engineering skills**.

---

## 🚀 Live Demo

👉 **Hugging Face Space:**
🔗 https://huggingface.co/spaces/hishaaaam/MLhousepredictor

Try the model directly in your browser — no setup required.

---

## ✨ Key Features

✅ Realistic housing price regression
✅ Advanced feature engineering
✅ Automated preprocessing pipeline
✅ Model comparison (Random Forest vs Gradient Boosting)
✅ Confidence interval estimation
✅ Premium blue glass UI
✅ Two-page professional dashboard
✅ Streamlit production deployment
✅ Industry-style project structure

---

# 📁 Project Structure

```
house-price-faang/
│
├── app.py
├── train_model.py
├── generate_data.py
├── requirements.txt
├── README.md
├── house_model.pkl        # generated after training
│
└── src/
    ├── __init__.py
    └── predictor.py
```

---

# ⚙️ Local Setup Instructions

## 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd house-price-faang
```

---

## 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# OR
source venv/bin/activate  # Mac/Linux
```

---

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Generate synthetic dataset

```bash
python generate_data.py
```

✔ Creates realistic housing data
✔ No external downloads required

---

## 5️⃣ Train the model

```bash
python train_model.py
```

✔ Performs model comparison
✔ Saves best model as `house_model.pkl`

---

## 6️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

# 🧠 ML Pipeline Overview

## 🔹 Feature Engineering

The model creates high-signal derived features:

* total_rooms
* area_per_room
* bath_bed_ratio
* luxury_indicator
* log_area

These mirror real-world real estate modeling practices.

---

## 🔹 Preprocessing

Implemented using sklearn `ColumnTransformer`:

* Numerical → StandardScaler
* Categorical → OneHotEncoder
* Unknown categories handled safely

---

## 🔹 Model Selection

The system automatically compares:

* Random Forest Regressor
* Gradient Boosting Regressor

Best model selected using **R² score**.

---

## 🔹 Prediction Confidence

Each prediction returns:

* Estimated Price
* Lower Bound (−10%)
* Upper Bound (+10%)

This simulates real-world uncertainty estimation.

---

# 🎨 Application Pages

## 🏠 Home Page

Displays:

* Project overview
* Pipeline explanation
* Evaluation metrics
* Residual performance plot

---

## 🤖 Prediction Page

Interactive inputs:

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Furnishing
* Amenities

Outputs:

* Predicted price
* Confidence range

---

# 📦 Requirements

```txt
streamlit==1.33.0
pandas
numpy
matplotlib
joblib
scikit-learn==1.4.2
shap
```

---

# 🏆 Resume-Ready Description

> Built an end-to-end house price prediction system using advanced feature engineering and ensemble regression models, deployed via a production-style Streamlit interface.

---

# 🔮 Future Improvements

* SHAP waterfall explainability
* XGBoost integration
* Real housing dataset
* Docker deployment
* REST API layer
* Model monitoring

---

## 👨‍💻 Author

**Hisham Hidayathulla**
Machine Learning • Data Science • AI Engineering

---

⭐ If this project helped you, consider giving it a star!
