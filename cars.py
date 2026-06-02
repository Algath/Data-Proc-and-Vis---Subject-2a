import pandas as pd
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "Cars_Datasets_2025.csv"
df = pd.read_csv(_DATA_PATH, encoding="cp1252")


def clean_column_names(df):
    fuel_type_dict = {
        "Petrol": "Petrol",
        "Diesel": "Diesel",
        "Hydrogen": "Hydrogen",
        "Electric": "Electric",
        "Hybrid": "Hybrid",
        # typo in source data
        "plug in hyrbrid": "Hybrid",
        # hybrid variants
        "Plug-in Hybrid": "Hybrid",
        "Hybrid / Plug-in": "Hybrid",
        "Petrol/Hybrid": "Hybrid",
        "Petrol, Hybrid": "Hybrid",
        "Petrol (Hybrid)": "Hybrid",
        "Hybrid (Petrol)": "Hybrid",
        "Hybrid/Petrol": "Hybrid",
        "Diesel Hybrid": "Hybrid",
        "Hybrid/Electric": "Hybrid",
        "Hybrid (Gas + Electric)": "Hybrid",
        "Gas / Hybrid": "Hybrid",
        "Petrol/EV": "Hybrid",
        # dominant fuel
        "Petrol/Diesel": "Petrol",
        "Petrol, Diesel": "Petrol",
        "Diesel/Petrol": "Diesel",
        "Petrol/AWD": "Petrol",  # AWD is drivetrain, not fuel
        # gas
        "CNG/Petrol": "Gas",
    }

    df["Fuel Types"] = df["Fuel Types"].map(fuel_type_dict)
    print(df["Fuel Types"].value_counts())
    print("Unmapped:", df["Fuel Types"].isna().sum())


def type_list(type_name):
    return df[df["Fuel Types"] == type_name]
