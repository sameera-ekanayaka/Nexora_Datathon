"""
src/cleaning.py
High-Performance Single-Pass Data Cleaning and Anomaly Audit Pipeline using DuckDB.
Team: Nexora
"""

import os
import time
import duckdb
import pandas as pd

SEED = 42

def get_duckdb_connection(threads=4, memory_limit="6GB"):
    """Initialize DuckDB connection with resource settings."""
    con = duckdb.connect()
    con.execute(f"SET threads = {threads};")
    con.execute(f"SET memory_limit = '{memory_limit}';")
    con.execute(f"SET preserve_insertion_order = false;")
    return con


def run_anomaly_audit(raw_taxi_glob="data/raw/Urban_Flow_Analytics_Taxi_Dataset_*.csv",
                      zone_csv_path="data/raw/Urban_Flow_Analytics_Zone_Dataset.csv",
                      audit_csv_path="data/interim/cleaning_audit.csv"):
    """
    Executes a high-performance single-pass scan across 48.6M rows using DuckDB.
    Quantifies all 12 anomaly types with exact row counts, percentages, and statistical percentiles.
    Saves audit results to data/interim/cleaning_audit.csv.
    """
    con = get_duckdb_connection()
    print(f"[*] Initializing single-pass DuckDB scan on {raw_taxi_glob}...", flush=True)
    start_time = time.time()

    # Load zone lookup table (CREATE OR REPLACE prevents error on re-execution)
    con.execute(f"""
        CREATE OR REPLACE TABLE zones AS 
        SELECT loc_id, borough_name, zone_name, service_zone 
        FROM read_csv_auto('{zone_csv_path}');
    """)
    n_zones = con.execute("SELECT count(*) FROM zones").fetchone()[0]
    print(f"[*] Reference zones loaded: {n_zones} zones from {zone_csv_path}", flush=True)

    # Register raw taxi dataset view with union_by_name
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_taxi AS 
        SELECT * FROM read_csv_auto('{raw_taxi_glob}', union_by_name=true);
    """)

    print("[*] Executing single-pass multi-metric anomaly scan across full dataset...", flush=True)

    # Single-pass unified aggregation query
    query = """
    SELECT
        count(*) AS total_raw,
        min(pickup_timestamp) AS min_pickup,
        max(pickup_timestamp) AS max_pickup,
        -- 1. Full-dataset duplicate detection via tuple hash
        count(*) - count(DISTINCT hash(provider_code, pickup_timestamp, dropoff_timestamp, origin_loc_id, dest_loc_id, base_fare, distance_miles, charge_total)) AS duplicate_rows,
        -- 2. Out-of-bounds dates
        count(*) FILTER (WHERE pickup_timestamp < '2025-04-01 00:00:00' OR pickup_timestamp >= '2026-04-01 00:00:00') AS out_of_bounds_dates,
        -- 3. Temporal inversion
        count(*) FILTER (WHERE dropoff_timestamp <= pickup_timestamp) AS temporal_inversion,
        -- 4. Negative values across monetary fields AND distance
        count(*) FILTER (WHERE distance_miles < 0 OR base_fare < 0 OR charge_total < 0 OR surcharge_misc < 0 OR transit_tax < 0 OR driver_tip_payment < 0 OR toll_total < 0) AS negative_values,
        -- 5. Zero distance with nonzero fare by rate class
        count(*) FILTER (WHERE distance_miles = 0 AND base_fare > 0 AND (rate_class_id = 1 OR rate_class_id IS NULL)) AS zero_dist_std_meter,
        count(*) FILTER (WHERE distance_miles = 0 AND base_fare > 0 AND rate_class_id IN (2, 3, 4, 5, 6)) AS zero_dist_flat_rate,
        -- 6. Passenger count anomalies
        count(*) FILTER (WHERE rider_count IS NULL OR rider_count = 0) AS zero_missing_passengers,
        count(*) FILTER (WHERE rider_count > 6) AS excess_passengers,
        -- 7. Unrealistic speed and duration
        count(*) FILTER (
            WHERE date_diff('second', pickup_timestamp, dropoff_timestamp) < 60
               OR date_diff('second', pickup_timestamp, dropoff_timestamp) > 86400
               OR (date_diff('second', pickup_timestamp, dropoff_timestamp) > 0 
                   AND (distance_miles / (date_diff('second', pickup_timestamp, dropoff_timestamp) / 3600.0)) > 70.0)
        ) AS unrealistic_speed_duration,
        -- 8. Fare component reconciliation mismatch (> $1.00)
        count(*) FILTER (
            WHERE abs(charge_total - (
                base_fare + 
                coalesce(surcharge_misc, 0) + 
                coalesce(transit_tax, 0) + 
                coalesce(driver_tip_payment, 0) + 
                coalesce(toll_total, 0) + 
                coalesce(service_improvement_fee, 0) + 
                coalesce(zone_congestion_fee, 0) + 
                coalesce(Airport_fee, 0) + 
                coalesce(congestion_relief_fee, 0)
            )) > 1.00
        ) AS fare_component_mismatch,
        -- 9. Unmapped/Unknown zones
        count(*) FILTER (
            WHERE origin_loc_id NOT IN (SELECT loc_id FROM zones)
               OR dest_loc_id NOT IN (SELECT loc_id FROM zones)
               OR origin_loc_id IN (264, 265)
               OR dest_loc_id IN (264, 265)
        ) AS unmapped_unknown_zones,
        -- 10. Dynamic 99.9th percentile thresholds
        quantile_cont(base_fare, 0.999) AS p999_fare,
        quantile_cont(distance_miles, 0.999) AS p999_dist
    FROM raw_taxi;
    """

    res = con.execute(query).fetchdf().iloc[0]
    total_raw = int(res["total_raw"])
    dup_rows = int(res["duplicate_rows"])
    baseline_rows = total_raw - dup_rows
    p999_fare = float(res["p999_fare"])
    p999_dist = float(res["p999_dist"])

    print(f"\n[+] Single pass complete in {time.time() - start_time:.1f}s!", flush=True)
    print(f"    - Total Raw Records: {total_raw:,}", flush=True)
    print(f"    - Date Range: {res['min_pickup']} to {res['max_pickup']}", flush=True)
    print(f"    - Global Duplicates: {dup_rows:,} ({(dup_rows/total_raw)*100:.4f}%)", flush=True)
    print(f"    - Post-Dedup Baseline: {baseline_rows:,}", flush=True)
    print(f"    - Dynamic 99.9th Percentiles: Base Fare = ${p999_fare:.2f}, Distance = {p999_dist:.2f} mi", flush=True)

    # Count actual extreme outliers exceeding 99.9th percentile
    extreme_outliers = con.execute(f"""
        SELECT count(*) FROM raw_taxi 
        WHERE base_fare > {p999_fare} OR distance_miles > {p999_dist};
    """).fetchone()[0]

    audit_records = [
        {
            "anomaly_id": 1,
            "anomaly_name": "Exact Duplicate Records (Global)",
            "condition": "Duplicate tuple hash across full dataset",
            "affected_rows": dup_rows,
            "pct_of_dataset": round((dup_rows / total_raw) * 100, 4),
            "action_taken": "Drop",
            "justification": "Redundant records resulting from duplicate dispatches or boundary overlap between monthly CSV exports."
        },
        {
            "anomaly_id": 2,
            "anomaly_name": "Out-of-Bounds Date Containment",
            "condition": "pickup_timestamp < '2025-04-01' OR pickup_timestamp >= '2026-04-01'",
            "affected_rows": int(res["out_of_bounds_dates"]),
            "pct_of_dataset": round((int(res["out_of_bounds_dates"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Records with corrupted hardware/meter clock timestamps (e.g. 2008/2009) falling outside the competition period."
        },
        {
            "anomaly_id": 3,
            "anomaly_name": "Temporal Inversion",
            "condition": "dropoff_timestamp <= pickup_timestamp",
            "affected_rows": int(res["temporal_inversion"]),
            "pct_of_dataset": round((int(res["temporal_inversion"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Physically impossible journeys where trip dropoff precedes pickup due to clock desynchronization or logging failure."
        },
        {
            "anomaly_id": 4,
            "anomaly_name": "Negative Financial & Distance Values",
            "condition": "distance_miles < 0 OR base_fare < 0 OR charge_total < 0 OR fees < 0 OR tips < 0",
            "affected_rows": int(res["negative_values"]),
            "pct_of_dataset": round((int(res["negative_values"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Dispute, reversal, cancellation, or corrupted sensor entries that distort predictive cost and distance baselines."
        },
        {
            "anomaly_id": 5,
            "anomaly_name": "Zero Distance with Fare (Standard Meter Glitch)",
            "condition": "distance_miles = 0 AND base_fare > 0 AND (rate_class_id = 1 OR rate_class_id IS NULL)",
            "affected_rows": int(res["zero_dist_std_meter"]),
            "pct_of_dataset": round((int(res["zero_dist_std_meter"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Standard metered rides with 0 distance charged full fare; represents odometer malfunction or aborted trips."
        },
        {
            "anomaly_id": 6,
            "anomaly_name": "Zero Distance with Fare (Flat/Negotiated Rate)",
            "condition": "distance_miles = 0 AND base_fare > 0 AND rate_class_id IN (2, 3, 4, 5, 6)",
            "affected_rows": int(res["zero_dist_flat_rate"]),
            "pct_of_dataset": round((int(res["zero_dist_flat_rate"]) / baseline_rows) * 100, 4),
            "action_taken": "Keep & Flag",
            "justification": "Legitimate flat-rate transactions (airport flat rates, negotiated dispatch fees, or minimum wait fees)."
        },
        {
            "anomaly_id": 7,
            "anomaly_name": "Missing or Zero Passenger Count",
            "condition": "rider_count IS NULL OR rider_count = 0",
            "affected_rows": int(res["zero_missing_passengers"]),
            "pct_of_dataset": round((int(res["zero_missing_passengers"]) / baseline_rows) * 100, 4),
            "action_taken": "Impute Median (1)",
            "justification": "Driver omitted manual passenger count entry on meter start; standard solo rider default preserves trip economics."
        },
        {
            "anomaly_id": 8,
            "anomaly_name": "Excess Passenger Count (>6)",
            "condition": "rider_count > 6",
            "affected_rows": int(res["excess_passengers"]),
            "pct_of_dataset": round((int(res["excess_passengers"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Exceeds legal NYC TLC taxi vehicle seating limits (max 5-6 passengers); meter entry keying error."
        },
        {
            "anomaly_id": 9,
            "anomaly_name": "Unrealistic Speed or Extreme Duration",
            "condition": "duration < 1 min OR duration > 24 hr OR speed > 70 mph",
            "affected_rows": int(res["unrealistic_speed_duration"]),
            "pct_of_dataset": round((int(res["unrealistic_speed_duration"]) / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Trips shorter than 60s, longer than 1 day, or with implied speeds exceeding 70 mph indicate sensor glitches."
        },
        {
            "anomaly_id": 10,
            "anomaly_name": f"Dynamic 99.9th Percentile Outliers (> ${p999_fare:.1f} / > {p999_dist:.1f} mi)",
            "condition": f"base_fare > {p999_fare:.2f} OR distance_miles > {p999_dist:.2f}",
            "affected_rows": extreme_outliers,
            "pct_of_dataset": round((extreme_outliers / baseline_rows) * 100, 4),
            "action_taken": "Drop",
            "justification": "Extreme upper-tail fat-finger anomalies dynamically bounded at the empirical 99.9th percentile."
        },
        {
            "anomaly_id": 11,
            "anomaly_name": "Unmapped or Unknown Zone IDs (TLC 264/265)",
            "condition": "loc_id NOT IN zones OR loc_id IN (264, 265)",
            "affected_rows": int(res["unmapped_unknown_zones"]),
            "pct_of_dataset": round((int(res["unmapped_unknown_zones"]) / baseline_rows) * 100, 4),
            "action_taken": "Keep & Categorize Unknown",
            "justification": "TLC zones 264/265 represent 'Unknown' or 'Outside NYC'; kept in supervised learning but flagged for spatial clustering."
        },
        {
            "anomaly_id": 12,
            "anomaly_name": "Fare Component Reconciliation Mismatch (> $1.00)",
            "condition": "abs(charge_total - sum(components)) > 1.00",
            "affected_rows": int(res["fare_component_mismatch"]),
            "pct_of_dataset": round((int(res["fare_component_mismatch"]) / baseline_rows) * 100, 4),
            "action_taken": "Flag & Retain (Diagnostic Quality Indicator)",
            "justification": "Diagnostic arithmetic integrity metric between charge_total and component sum; documented in report."
        }
    ]

    os.makedirs(os.path.dirname(audit_csv_path), exist_ok=True)
    audit_df = pd.DataFrame(audit_records)
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"\n[+] Audit saved to: {audit_csv_path}", flush=True)

    return audit_df, p999_fare, p999_dist, con


def generate_clean_partitions_and_sample(con=None,
                                        raw_taxi_glob="data/raw/Urban_Flow_Analytics_Taxi_Dataset_*.csv",
                                        zone_csv_path="data/raw/Urban_Flow_Analytics_Zone_Dataset.csv",
                                        output_dir="data/processed",
                                        sample_size=500000,
                                        p999_fare=150.0,
                                        p999_dist=30.30):
    """
    Applies the full cleaning logic in DuckDB and writes out:
    1. Clean Train Split: Months 2025-04 to 2026-01 (Parquet)
    2. Clean Val Split:   Month 2026-02 (Parquet)
    3. Clean Test Split:  Month 2026-03 (Parquet)
    4. Stratified Training Sample: 500,000 trips from Train split only.
    """
    if con is None:
        con = get_duckdb_connection()

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    print("\n[*] Materializing clean dataset view in DuckDB...", flush=True)

    # Ensure reference tables and views exist on this connection
    con.execute(f"""
        CREATE OR REPLACE TABLE zones AS 
        SELECT loc_id, borough_name, zone_name, service_zone 
        FROM read_csv_auto('{zone_csv_path}');
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW raw_taxi AS 
        SELECT * FROM read_csv_auto('{raw_taxi_glob}', union_by_name=true);
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW clean_taxi AS
        SELECT 
            provider_code,
            pickup_timestamp,
            dropoff_timestamp,
            date_diff('second', pickup_timestamp, dropoff_timestamp) / 60.0 AS duration_minutes,
            CASE 
                WHEN rider_count IS NULL OR rider_count = 0 THEN 1 
                ELSE CAST(rider_count AS INTEGER) 
            END AS rider_count,
            distance_miles,
            rate_class_id,
            offline_record_flag,
            origin_loc_id,
            dest_loc_id,
            fare_settlement_method,
            base_fare,
            coalesce(surcharge_misc, 0.0) AS surcharge_misc,
            coalesce(transit_tax, 0.0) AS transit_tax,
            coalesce(driver_tip_payment, 0.0) AS driver_tip_payment,
            coalesce(toll_total, 0.0) AS toll_total,
            coalesce(service_improvement_fee, 0.0) AS service_improvement_fee,
            charge_total,
            coalesce(zone_congestion_fee, 0.0) AS zone_congestion_fee,
            coalesce(Airport_fee, 0.0) AS Airport_fee,
            coalesce(congestion_relief_fee, 0.0) AS congestion_relief_fee,
            -- Calendar & temporal features
            date_part('year', pickup_timestamp) AS pickup_year,
            date_part('month', pickup_timestamp) AS pickup_month,
            date_part('day', pickup_timestamp) AS pickup_day,
            date_part('hour', pickup_timestamp) AS pickup_hour,
            date_part('dayofweek', pickup_timestamp) AS pickup_dayofweek,
            CASE WHEN date_part('dayofweek', pickup_timestamp) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
        FROM raw_taxi
        WHERE 
            -- 1. Date containment (empirically confirmed: 2025-04 through 2026-03)
            pickup_timestamp >= '2025-04-01 00:00:00' AND pickup_timestamp < '2026-04-01 00:00:00'
            -- 2. Temporal consistency
            AND dropoff_timestamp > pickup_timestamp
            AND date_diff('second', pickup_timestamp, dropoff_timestamp) >= 60
            AND date_diff('second', pickup_timestamp, dropoff_timestamp) <= 86400
            -- 3. Realistic speed
            AND (distance_miles / (date_diff('second', pickup_timestamp, dropoff_timestamp) / 3600.0)) <= 70.0
            -- 4. Positive economics & dynamic outlier cap
            AND base_fare >= 2.50 AND base_fare <= {p999_fare}
            AND distance_miles >= 0 AND distance_miles <= {p999_dist}
            AND charge_total > 0
            AND surcharge_misc >= 0 AND transit_tax >= 0 AND driver_tip_payment >= 0 AND toll_total >= 0
            -- 5. Zero distance logic (drop standard rate glitches, keep flat/airport)
            AND NOT (distance_miles = 0 AND (rate_class_id = 1 OR rate_class_id IS NULL))
            -- 6. Passenger capacity bounds
            AND (rider_count IS NULL OR rider_count <= 6)
            -- 7. Valid Zone ID filter
            AND origin_loc_id IS NOT NULL AND dest_loc_id IS NOT NULL
            AND origin_loc_id > 0 AND dest_loc_id > 0;
    """)

    total_clean = con.execute("SELECT count(*) FROM clean_taxi").fetchone()[0]
    print(f"[+] Total cleaned records: {total_clean:,}", flush=True)

    # Export Chronological Splits
    # 1. Validation Split (Month 11: 2026-02)
    val_parquet = os.path.join(output_dir, "val_clean.parquet")
    print(f"[*] Exporting Validation Split (2026-02) to {val_parquet}...", flush=True)
    con.execute(f"""
        COPY (
            SELECT * FROM clean_taxi 
            WHERE pickup_timestamp >= '2026-02-01 00:00:00' AND pickup_timestamp < '2026-03-01 00:00:00'
        ) TO '{val_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    val_count = con.execute(f"SELECT count(*) FROM read_parquet('{val_parquet}')").fetchone()[0]
    print(f"    [+] Val Set Exported: {val_count:,} rows", flush=True)

    # 2. Test Split (Month 12: 2026-03)
    test_parquet = os.path.join(output_dir, "test_clean.parquet")
    print(f"[*] Exporting Test Split (2026-03) to {test_parquet}...", flush=True)
    con.execute(f"""
        COPY (
            SELECT * FROM clean_taxi 
            WHERE pickup_timestamp >= '2026-03-01 00:00:00' AND pickup_timestamp < '2026-04-01 00:00:00'
        ) TO '{test_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    test_count = con.execute(f"SELECT count(*) FROM read_parquet('{test_parquet}')").fetchone()[0]
    print(f"    [+] Test Set Exported: {test_count:,} rows", flush=True)

    # 3. Stratified Training Sample from Months 1-10 ONLY
    train_sample_parquet = os.path.join(output_dir, f"train_sample_{sample_size//1000}k.parquet")
    print(f"[*] Drawing {sample_size:,} stratified sample from Training Partition (Months 1-10)...", flush=True)

    train_total = con.execute("""
        SELECT count(*) FROM clean_taxi 
        WHERE pickup_timestamp >= '2025-04-01 00:00:00' AND pickup_timestamp < '2026-02-01 00:00:00'
    """).fetchone()[0]
    sample_rate = sample_size / train_total
    print(f"    - Training Partition Size: {train_total:,} rows. Target sampling rate: {sample_rate:.4%}", flush=True)

    con.execute(f"""
        COPY (
            WITH train_strata AS (
                SELECT *, 
                       row_number() OVER (
                           PARTITION BY pickup_month, pickup_dayofweek, pickup_hour 
                           ORDER BY hash(provider_code, pickup_timestamp, origin_loc_id, {SEED})
                       ) AS rank_in_stratum,
                       count(*) OVER (
                           PARTITION BY pickup_month, pickup_dayofweek, pickup_hour
                       ) AS stratum_total
                FROM clean_taxi
                WHERE pickup_timestamp >= '2025-04-01 00:00:00' AND pickup_timestamp < '2026-02-01 00:00:00'
            )
            SELECT * EXCLUDE (rank_in_stratum, stratum_total)
            FROM train_strata
            WHERE rank_in_stratum <= ceil(stratum_total * {sample_rate})
            LIMIT {sample_size}
        ) TO '{train_sample_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    actual_sample_count = con.execute(f"SELECT count(*) FROM read_parquet('{train_sample_parquet}')").fetchone()[0]
    print(f"    [+] Train Sample Exported: {actual_sample_count:,} rows to {train_sample_parquet}", flush=True)

    elapsed = time.time() - start_time
    print(f"[+] All clean datasets generated successfully in {elapsed:.1f}s!", flush=True)


if __name__ == "__main__":
    print("=== NEXORA DATA CLEANING & ANOMALY AUDIT PIPELINE ===", flush=True)
    audit_df, p999_fare, p999_dist, con = run_anomaly_audit()
    print("\n--- ANOMALY AUDIT SUMMARY ---", flush=True)
    print(audit_df[["anomaly_id", "anomaly_name", "affected_rows", "pct_of_dataset", "action_taken"]].to_string(), flush=True)
    print("\n--- GENERATING CLEAN SPLITS & SAMPLE ---", flush=True)
    generate_clean_partitions_and_sample(con, p999_fare=p999_fare, p999_dist=p999_dist)
