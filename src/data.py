# Handles loading data, filling missing values, and cleaning outliers

import numpy as np
import pandas as pd


def load_dataset(path):
    """Loads the CSV file and prints a quick summary of whats inside."""
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} water samples with {df.shape[1]} columns")
    print(f"Class distribution:\n{df['Potability'].value_counts()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"Missing values found:\n{missing}")
    return df


def fit_imputer(df_train):
    """Learns the median values from training data for each class separately."""
    cols_with_missing = df_train.columns[df_train.isnull().any()].tolist()
    if "Potability" in cols_with_missing:
        cols_with_missing.remove("Potability")

    medians = {}
    for col in cols_with_missing:
        medians[col] = {}
        for label in [0, 1]:
            medians[col][label] = float(
                df_train.loc[df_train["Potability"] == label, col].median()
            )

    print(f"Computed medians for {len(cols_with_missing)} columns: {cols_with_missing}")
    return medians


def apply_imputer(df, medians):
    """Fills in missing values using the pre-computed medians."""
    df = df.copy()
    for col, class_medians in medians.items():
        for label, med_val in class_medians.items():
            mask = (df["Potability"] == label) & (df[col].isnull())
            df.loc[mask, col] = med_val

    # Safety fallback for any remaining nulls
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        for col in df.columns:
            if df[col].isnull().any() and col != "Potability":
                df[col] = df[col].fillna(df[col].median())

    assert df.isnull().sum().sum() == 0, "Still have missing values!"
    return df


def fit_outlier_clipper(df_train, lower_pct=0.01, upper_pct=0.99):
    """Finds the 1st and 99th percentile boundaries from training data."""
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    if "Potability" in numeric_cols:
        numeric_cols.remove("Potability")

    bounds = {}
    for col in numeric_cols:
        low = float(df_train[col].quantile(lower_pct))
        high = float(df_train[col].quantile(upper_pct))
        bounds[col] = (low, high)

    print(f"Computed outlier bounds for {len(numeric_cols)} columns")
    return bounds


def apply_outlier_clipper(df, bounds):
    """Clips extreme values to stay within the learned boundaries."""
    df = df.copy()
    total_clipped = 0
    for col, (low, high) in bounds.items():
        if col in df.columns:
            outliers = ((df[col] < low) | (df[col] > high)).sum()
            df[col] = df[col].clip(lower=low, upper=high)
            total_clipped += outliers

    print(f"Clipped {total_clipped} outlier values")
    return df
