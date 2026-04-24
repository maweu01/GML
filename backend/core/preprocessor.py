"""
Guardian ML — Data Preprocessor & Feature Engineer
Handles CSV/JSON ingestion, validation, cleaning, and feature engineering.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.logger import get_logger

logger = get_logger("preprocessor")


class DataPreprocessor:
    """
    Full preprocessing pipeline:
      1. Load data (CSV or JSON)
      2. Validate schema
      3. Clean nulls / outliers
      4. Encode categorical features
      5. Scale numeric features
      6. Return feature matrix X and optional target y
    """

    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self.target_col: Optional[str] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        logger.info(f"Loaded dataset: {path} — shape {df.shape}")
        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit preprocessor and transform data. Returns (X, y)."""
        self.target_col = target_col
        df = self._clean(df)
        df = self._encode_categoricals(df, fit=True)
        X, y = self._split_features_target(df, target_col)
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)
        self._fitted = True
        logger.info(f"Fit-transform complete. Features: {len(self.feature_names)}, Samples: {X_scaled.shape[0]}")
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")
        df = self._clean(df)
        df = self._encode_categoricals(df, fit=False)
        # Align columns to fitted feature set
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names]
        return self.scaler.transform(df)

    def feature_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for feature reporting."""
        numeric = df.select_dtypes(include=[np.number])
        return {
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1]),
            "numeric_features": list(numeric.columns),
            "categorical_features": list(df.select_dtypes(include=["object", "category"]).columns),
            "null_counts": df.isnull().sum().to_dict(),
            "statistics": numeric.describe().to_dict(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop fully-empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        # Fill numeric nulls with median
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        # Fill categorical nulls with mode
        for col in df.select_dtypes(include=["object", "category"]).columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else "unknown")
        logger.debug(f"Cleaned DataFrame: {df.shape}")
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col == self.target_col:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
                else:
                    df[col] = 0
        return df

    def _split_features_target(
        self, df: pd.DataFrame, target_col: Optional[str]
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y_raw = df[target_col]
            # Encode target if categorical
            if y_raw.dtype == object or str(y_raw.dtype) == "category":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                self.label_encoders["__target__"] = le
            else:
                y = y_raw.to_numpy()
            return X, y
        return df, None
