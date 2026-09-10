"""
src/features.py
Feature Engineering Pipeline with Strict Pre-Trip Leakage Discipline.
Team: Nexora
"""

import numpy as np
import pandas as pd

SEED = 42

# Major Airport Location IDs in TLC system:
# 1: Newark (EWR), 132: JFK Airport, 138: LaGuardia Airport (LGA)
AIRPORT_LOC_IDS = {1, 132, 138}

# Congestion / Commercial Core Zones (Midtown, Lower Manhattan, Financial District)
MANHATTAN_CORE_ZONES = {
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107,
    113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148,
    151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209,
    211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244,
    246, 249, 261, 262, 263
}


def build_train_od_lookup(train_df):
    """
    Computes historical mean distance and duration per (origin_loc_id, dest_loc_id)
    STRICTLY on the training fold.
    Returns:
      od_lookup: DataFrame with (origin_loc_id, dest_loc_id, od_mean_distance, od_mean_duration, od_trip_count)
      global_stats: dict with global mean distance and duration for unseen pairs
    """
    print("[*] Computing historical OD lookup table from Training Fold...")
    od_stats = train_df.groupby(["origin_loc_id", "dest_loc_id"]).agg(
        od_mean_distance=("distance_miles", "mean"),
        od_mean_duration=("duration_minutes", "mean"),
        od_trip_count=("distance_miles", "count")
    ).reset_index()

    global_stats = {
        "global_mean_distance": float(train_df["distance_miles"].mean()),
        "global_mean_duration": float(train_df["duration_minutes"].mean()),
        "global_median_distance": float(train_df["distance_miles"].median()),
        "global_median_duration": float(train_df["duration_minutes"].median())
    }

    print(f"    [+] Computed stats for {len(od_stats):,} unique OD pairs.")
    print(f"    [+] Global defaults: Distance = {global_stats['global_mean_distance']:.2f} mi, "
          f"Duration = {global_stats['global_mean_duration']:.2f} min")
    return od_stats, global_stats


def apply_od_lookup(df, od_lookup, global_stats):
    """
    Left-joins the frozen OD lookup table to target dataset and imputes unseen pairs
    using global training fold defaults.
    """
    merged = df.merge(od_lookup, on=["origin_loc_id", "dest_loc_id"], how="left")
    
    # Impute unseen OD pairs with global training defaults
    merged["od_mean_distance"] = merged["od_mean_distance"].fillna(global_stats["global_mean_distance"])
    merged["od_mean_duration"] = merged["od_mean_duration"].fillna(global_stats["global_mean_duration"])
    merged["od_trip_count"] = merged["od_trip_count"].fillna(0).astype(int)
    
    return merged


def engineer_pretrip_features(df, is_training=True):
    """
    Constructs clean, leak-free pre-trip features available before trip departure.
    """
    feat = df.copy()

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(feat["pickup_timestamp"]):
        feat["pickup_timestamp"] = pd.to_datetime(feat["pickup_timestamp"])

    # Temporal features
    feat["pickup_hour"] = feat["pickup_timestamp"].dt.hour
    feat["pickup_dayofweek"] = feat["pickup_timestamp"].dt.dayofweek
    feat["pickup_month"] = feat["pickup_timestamp"].dt.month
    feat["pickup_day"] = feat["pickup_timestamp"].dt.day
    feat["is_weekend"] = feat["pickup_dayofweek"].isin([5, 6]).astype(int)

    # Cyclical hour encoding
    feat["sin_hour"] = np.sin(2 * np.pi * feat["pickup_hour"] / 24.0)
    feat["cos_hour"] = np.cos(2 * np.pi * feat["pickup_hour"] / 24.0)

    # Cyclical day of week encoding
    feat["sin_dow"] = np.sin(2 * np.pi * feat["pickup_dayofweek"] / 7.0)
    feat["cos_dow"] = np.cos(2 * np.pi * feat["pickup_dayofweek"] / 7.0)

    # Rush-hour indicators:
    # Morning rush: 07:00–10:00 (Weekdays)
    # Evening rush: 16:00–20:00 (Weekdays)
    feat["is_morning_rush"] = ((feat["is_weekend"] == 0) & (feat["pickup_hour"].between(7, 9))).astype(int)
    feat["is_evening_rush"] = ((feat["is_weekend"] == 0) & (feat["pickup_hour"].between(16, 19))).astype(int)
    feat["is_night"] = (feat["pickup_hour"].between(22, 23) | feat["pickup_hour"].between(0, 4)).astype(int)

    # Spatial context indicators
    feat["is_origin_airport"] = feat["origin_loc_id"].isin(AIRPORT_LOC_IDS).astype(int)
    feat["is_dest_airport"] = feat["dest_loc_id"].isin(AIRPORT_LOC_IDS).astype(int)
    feat["is_airport_trip"] = (feat["is_origin_airport"] | feat["is_dest_airport"]).astype(int)

    feat["is_origin_core"] = feat["origin_loc_id"].isin(MANHATTAN_CORE_ZONES).astype(int)
    feat["is_dest_core"] = feat["dest_loc_id"].isin(MANHATTAN_CORE_ZONES).astype(int)
    feat["is_core_to_core"] = (feat["is_origin_core"] & feat["is_dest_core"]).astype(int)

    # Intra-zone trip flag
    feat["is_same_zone"] = (feat["origin_loc_id"] == feat["dest_loc_id"]).astype(int)

    # Rate Class ID Pre-Trip Flag:
    # Standard rate (1) vs Flat Airport rate (2) vs other
    feat["is_airport_flat_rate"] = (feat["rate_class_id"] == 2).astype(int)

    # Categorical features cast to category dtype
    cat_cols = ["origin_loc_id", "dest_loc_id", "provider_code"]
    for col in cat_cols:
        if col in feat.columns:
            feat[col] = feat[col].astype("category")

    return feat


def get_feature_columns():
    """
    Returns the strict list of pre-trip features permitted for model training.
    Guaranteed zero post-trip leakage!
    """
    return [
        "origin_loc_id",
        "dest_loc_id",
        "provider_code",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "sin_dow",
        "cos_dow",
        "is_morning_rush",
        "is_evening_rush",
        "is_night",
        "is_origin_airport",
        "is_dest_airport",
        "is_airport_trip",
        "is_origin_core",
        "is_dest_core",
        "is_core_to_core",
        "is_same_zone",
        "is_airport_flat_rate",
        "od_mean_distance",
        "od_mean_duration",
        "od_trip_count"
    ]
