import joblib
import pandas as pd
import numpy as np

model = joblib.load("house_model.pkl")


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


def predict_price(input_dict):
    df = pd.DataFrame([input_dict])
    df = engineer_features(df)

    pred = model.predict(df)[0]

    # simple confidence interval
    lower = pred * 0.9
    upper = pred * 1.1

    return round(pred, 0), round(lower, 0), round(upper, 0)