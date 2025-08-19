# preprocessing.py
from __future__ import annotations
import re
from typing import Optional
from math import isnan

import numpy as np
import pandas as pd

# Unit conversion to sqft (approx)
UNIT_TO_SQFT = {
    "sq. meter": 10.7639,
    "sq meter": 10.7639,
    "sqm": 10.7639,
    "acre": 43560.0,
    "acres": 43560.0,
    "hectare": 107639.0,
    "cent": 435.6,
    "guntha": 1089.0,
    "perch": 272.25,
    "sq yard": 9.0,      # 1 sq yard = 9 sqft
    "sq. yard": 9.0,
    "gaj": 9.0
}

RANGE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", re.I)

def _extract_first_float(s: str) -> Optional[float]:
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except:
        return None

def parse_total_sqft(val) -> Optional[float]:
    """
    Convert total_sqft entries to float (sqft). Handles:
    - numeric: "1200"
    - ranges: "1000 - 1200" -> mean
    - with units: "34.46 Sq. Meter", "2.5 acres" -> convert to sqft
    Returns None if unparseable.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    # Range: take mean
    m = RANGE_PATTERN.match(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return (a + b) / 2.0

    # Try recognized units
    for unit, factor in UNIT_TO_SQFT.items():
        if unit in s:
            num = _extract_first_float(s)
            return float(num) * factor if num is not None else None

    # Strip common sqft tokens
    s = (s.replace("sqft", "")
           .replace("sq. ft", "")
           .replace("sq ft", "")
           .strip())
    try:
        return float(s)
    except:
        num = _extract_first_float(s)
        return float(num) if num is not None else None

def ensure_bhk(df: pd.DataFrame) -> pd.DataFrame:
    """Create/normalize 'bhk' from 'size' if needed (e.g., '2 BHK')."""
    df = df.copy()
    if "bhk" not in df.columns:
        if "size" in df.columns:
            df["bhk"] = df["size"].astype(str).str.extract(r"(\d+)").astype(float)
        else:
            df["bhk"] = np.nan
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight, non-destructive feature engineering."""
    df = df.copy()

    # total_sqft numeric
    if "total_sqft" in df.columns:
        df["total_sqft"] = df["total_sqft"].apply(parse_total_sqft)

    # bhk
    df = ensure_bhk(df)

    # area_per_bhk (safe division)
    if "total_sqft" in df.columns and "bhk" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["area_per_bhk"] = df["total_sqft"] / df["bhk"].replace({0: np.nan})

    return df

def basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only plausible values but be gentle to avoid dropping everything."""
    df = df.copy()
    if "total_sqft" in df.columns:
        df = df[(df["total_sqft"].notna()) & (df["total_sqft"] >= 200) & (df["total_sqft"] <= 20000)]
    if "price" in df.columns:
        df = df[(df["price"].notna()) & (df["price"] > 0)]
    if "bhk" in df.columns:
        df = df[(df["bhk"].notna()) & (df["bhk"] >= 1) & (df["bhk"] <= 12)]
    # Don't drop on 'bath' here; we'll impute later.
    if "location" in df.columns:
        df["location"] = df["location"].astype(str).str.strip()
        df = df[df["location"] != ""]
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning used by training:
      1) Add features
      2) Gentle filters
      3) Drop rows missing just the CRITICAL fields we truly need for training
    Kaggle columns: area_type, availability, location, size, society, total_sqft, bath, balcony, price
    """
    df = add_features(df)
    df = basic_filters(df)
    # Critical fields for training
    critical = [c for c in ["total_sqft", "bhk", "price", "location"] if c in df.columns]
    df = df.dropna(subset=critical)
    return df
