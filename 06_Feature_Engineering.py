"""
Feature engineering for the "Predicting Electric Vehicle Purchases"
(Playground Series S6E9) dataset.

Findings from 01_EDA.ipynb that drove these features:
  - No missing values, no duplicates, no train/test distribution shift
    (KS tests all non-significant) -> we can engineer freely without
    worrying about leakage from imputation or shift correction.
  - Target is imbalanced: ~17.5% positive (Will_Buy_EV == 1).
  - Strongest single-feature signal (mutual information):
        Subsidy_Available            (cat, MI 0.162, Cramer's V 0.342)
        Environmental_Concern_Level  (num, MI 0.148, |Pearson| 0.46)
        Home_Charging_Possible       (cat, MI 0.100)
        Range_Anxiety_Level          (cat, MI 0.080, ordinal: Low>Medium>High
                                       purchase rate collapses from 18.9%->0.14%)
        Annual_Income_USD            (num, MI 0.054)
        City_Type                    (cat, MI 0.051)
  - Weak signal: Charging_Stations_Near_Home/Work, Daily_Commute_km, Age,
    Number_of_Cars_Owned individually, but they may still help in
    interactions/ratios and in tree-based models.
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
TRAIN_FE_PATH = "./dataset/train_fe.csv"
TEST_FE_PATH = "./dataset/test_fe.csv"

TARGET = "Will_Buy_EV"
ID_COL = "id"

NUM_COLS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CAT_COLS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

RANGE_ANXIETY_MAP = {"Low": 0, "Medium": 1, "High": 2}
YES_NO_MAP = {"Yes": 1, "No": 0}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_raw_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """Load raw train/test csv files and map the target to 0/1."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if train[TARGET].dtype == object:
        train[TARGET] = train[TARGET].map(YES_NO_MAP).astype(int)

    return train, test


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features on top of the raw columns.
    Safe to call on train and test separately (no target leakage;
    everything here is row-wise / deterministic).
    """
    df = df.copy()

    # --- binary-encoded flags (kept alongside the original string cols,
    #     useful for numeric interactions and for the linear baseline) ---
    df["Home_Charging_bin"] = df["Home_Charging_Possible"].map(YES_NO_MAP)
    df["Subsidy_bin"] = df["Subsidy_Available"].map(YES_NO_MAP)
    df["Range_Anxiety_ord"] = df["Range_Anxiety_Level"].map(RANGE_ANXIETY_MAP)

    # --- ratios / interactions on numeric columns ---
    df["Income_per_Age"] = df["Annual_Income_USD"] / (df["Age"] + 1)
    df["Commute_per_Car"] = df["Daily_Commute_km"] / (df["Number_of_Cars_Owned"] + 1)

    df["Total_Charging_Stations"] = (
        df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
    )
    df["Charging_Station_Ratio"] = df["Charging_Stations_Near_Home"] / (
        df["Charging_Stations_Near_Work"] + 1
    )

    # Access score: only "counts" if the person can actually charge at home
    df["Charging_Access_Score"] = (
        df["Total_Charging_Stations"] * df["Home_Charging_bin"]
    )

    # --- interactions with the two strongest signals (Subsidy x EnvConcern) ---
    df["Subsidy_x_EnvConcern"] = df["Subsidy_bin"] * df["Environmental_Concern_Level"]
    df["Subsidy_x_HomeCharging"] = df["Subsidy_bin"] * df["Home_Charging_bin"]
    df["EnvConcern_minus_RangeAnxiety"] = (
        df["Environmental_Concern_Level"] - df["Range_Anxiety_ord"]
    )

    # --- simple flags ---
    df["High_Env_Concern"] = (df["Environmental_Concern_Level"] >= 4).astype(int)
    df["Long_Commute"] = (df["Daily_Commute_km"] >= df["Daily_Commute_km"].median()).astype(int)

    # --- binned age group (ordinal-encoded, avoids adding more cat columns) ---
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=False,
    )

    return df


def get_feature_lists(df: pd.DataFrame):
    """
    Return (num_cols, cat_cols) after engineering, based on which
    engineered columns exist in `df`. Original raw cat columns are kept
    as categorical (for CatBoost/LightGBM); engineered binary/ordinal
    features are treated as numeric.
    """
    engineered_num = [
        "Home_Charging_bin",
        "Subsidy_bin",
        "Range_Anxiety_ord",
        "Income_per_Age",
        "Commute_per_Car",
        "Total_Charging_Stations",
        "Charging_Station_Ratio",
        "Charging_Access_Score",
        "Subsidy_x_EnvConcern",
        "Subsidy_x_HomeCharging",
        "EnvConcern_minus_RangeAnxiety",
        "High_Env_Concern",
        "Long_Commute",
        "Age_Group",
    ]
    num_cols = NUM_COLS + [c for c in engineered_num if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    return num_cols, cat_cols


# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
def main():
    os.makedirs("./dataset", exist_ok=True)

    train, test = load_raw_data()
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    num_cols, cat_cols = get_feature_lists(train_fe)

    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Original columns : {train.shape[1]}")
    print(f"Engineered columns: {train_fe.shape[1]}")
    print(f"Numeric features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    train_fe.to_csv(TRAIN_FE_PATH, index=False)
    test_fe.to_csv(TEST_FE_PATH, index=False)
    print(f"\nSaved: {TRAIN_FE_PATH}")
    print(f"Saved: {TEST_FE_PATH}")


if __name__ == "__main__":
    main()