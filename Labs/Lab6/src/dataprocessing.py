# dataprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def feature_transform(df):
    # Drop ID as it's not useful
    df = df.drop(columns=["id"])

    if "Price" in df.columns:
        X = df.drop("Price", axis=1)
        y = df["Price"]
    else:
        X = df
        y = None

    # Identify numeric and categorical columns
    numeric_features = ["Compartments", "Weight Capacity (kg)"]
    categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y
