import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ================= LOAD =================
df = pd.read_csv("housing.csv")


# ================= FEATURE ENGINEERING =================
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


df = engineer_features(df)

TARGET = "price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ================= MODEL COMPARISON =================
models = {
    "rf": RandomForestRegressor(n_estimators=300, random_state=42),
    "gb": GradientBoostingRegressor(random_state=42),
}

best_model = None
best_score = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, m in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", m)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"{name} R2:", score)

    if score > best_score:
        best_score = score
        best_model = pipe

joblib.dump(best_model, "house_model.pkl")
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")

print("✅ Best model saved")