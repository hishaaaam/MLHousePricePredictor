import pandas as pd
import numpy as np

np.random.seed(42)
n = 800

data = {
    "area": np.random.randint(400, 6000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "stories": np.random.randint(1, 4, n),
    "parking": np.random.randint(0, 3, n),
    "furnishingstatus": np.random.choice(
        ["furnished", "semi-furnished", "unfurnished"], n
    ),
    "mainroad": np.random.choice(["yes", "no"], n),
    "guestroom": np.random.choice(["yes", "no"], n),
    "basement": np.random.choice(["yes", "no"], n),
    "hotwaterheating": np.random.choice(["yes", "no"], n),
    "airconditioning": np.random.choice(["yes", "no"], n),
}

df = pd.DataFrame(data)

# ===== realistic price function =====
base_price = (
    df["area"] * 320
    + df["bedrooms"] * 60000
    + df["bathrooms"] * 45000
    + df["stories"] * 50000
    + df["parking"] * 35000
)

bonus = (
    (df["airconditioning"] == "yes") * 80000
    + (df["mainroad"] == "yes") * 60000
    + (df["guestroom"] == "yes") * 40000
)

noise = np.random.normal(0, 80000, n)

df["price"] = (base_price + bonus + noise).astype(int)

df.to_csv("housing.csv", index=False)
print("✅ Realistic housing.csv generated")