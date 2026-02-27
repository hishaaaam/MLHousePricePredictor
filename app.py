import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from predictor import predict_price
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="House Price AI",
    layout="wide",
    page_icon="🏠"
)

# =====================================================
# PREMIUM BLUE GLASS THEME
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.glass {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.big-title {
    font-size: 48px;
    font-weight: 700;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🤖 Predict Price"]
)

# =====================================================
# CACHED MODEL LOADING
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

# =====================================================
# FEATURE ENGINEERING (same as training)
# =====================================================
def engineer_features(df):
    df = df.copy()
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["area_per_room"] = df["area"] / (df["total_rooms"] + 1)
    df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1)
    df["is_luxury"] = (
        (df["area"] > 4000)
        & (df["airconditioning"] == "yes")
        & (df["parking"] >= 2)
    ).astype(int)
    df["log_area"] = np.log1p(df["area"])
    return df

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.markdown('<div class="big-title">🏠 AI House Price Predictor</div>', unsafe_allow_html=True)
    st.caption("End-to-End FAANG-Style Regression System")

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📌 About the Model")

    st.write("""
This system predicts house prices using an advanced machine learning pipeline.

**Pipeline highlights:**
- ✔ Feature engineering (structural + luxury indicators)  
- ✔ Automated preprocessing pipeline  
- ✔ Model comparison (Random Forest vs Gradient Boosting)  
- ✔ Hyperparameter tuning  
- ✔ Confidence interval estimation  
- ✔ Production-ready deployment  
""")

    st.markdown('</div>', unsafe_allow_html=True)

    # ================= METRICS =================
    st.subheader("📊 Model Evaluation")

    try:
        df = pd.read_csv("housing.csv")
        df = engineer_features(df)

        X = df.drop(columns=["price"])
        y = df["price"]

        preds = model.predict(X)

        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("R² Score", f"{r2:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("MAE", f"₹ {mae:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RMSE", f"₹ {rmse:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== residual plot =====
        fig, ax = plt.subplots()
        ax.scatter(y, preds, alpha=0.4)
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    except Exception as e:
        st.warning("Metrics will appear after housing.csv is available.")

# =====================================================
# PREDICTION PAGE
# =====================================================
else:

    st.title("🤖 Predict House Price")

    with st.container():
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            area = st.number_input("Area (sq ft)", 300, 10000, 1500)
            bedrooms = st.slider("Bedrooms", 1, 6, 3)
            bathrooms = st.slider("Bathrooms", 1, 5, 2)

        with col2:
            stories = st.slider("Stories", 1, 4, 2)
            parking = st.slider("Parking", 0, 3, 1)
            furnishing = st.selectbox(
                "Furnishing",
                ["furnished", "semi-furnished", "unfurnished"]
            )

        with col3:
            mainroad = st.selectbox("Main Road", ["yes", "no"])
            guestroom = st.selectbox("Guest Room", ["yes", "no"])
            basement = st.selectbox("Basement", ["yes", "no"])
            hotwater = st.selectbox("Hot Water Heating", ["yes", "no"])
            ac = st.selectbox("Air Conditioning", ["yes", "no"])

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= PREDICT =================
    @st.cache_data
    def cached_predict(data):
        return predict_price(data)

    if st.button("🚀 Predict Price", use_container_width=True):
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "furnishingstatus": furnishing,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwater,
            "airconditioning": ac,
        }

        pred, low, high = cached_predict(input_data)

        st.success(f"💰 Estimated Price: ₹ {pred:,.0f}")
        st.info(f"📊 Confidence Range: ₹ {low:,.0f} — ₹ {high:,.0f}")