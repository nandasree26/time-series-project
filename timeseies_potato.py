"""
Corrected Instrumented Time Series Pipeline for merged_potato_reservoir.csv
- Keeps original steps 1..9 but fixes:
  * canonical state grouping to avoid duplicate state folders
  * calendar-year train/test split (train = first..last-TEST_YEARS, test = last-TEST_YEARS+1..last)
  * always save per-state summary and outputs (even when model not trained)
  * ensure stationarity/differencing steps executed and logged for every state
  * robust ARIMAX handling and exog_future construction
Author: (Your Name)
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "merged_potato_reservoir.csv"
OUTPUT_ROOT = "outputs"
FORECAST_YEARS = [2023, 2024, 2025]      # explicit requested forecast years
TEST_YEARS = 2                           # last N calendar years used for test
MIN_POINTS_FOR_MODEL = 5                 # minimum yearly points to try modeling
# earliest year to include in training (None to use earliest available)
MIN_TRAIN_YEAR = 2000
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------------------
# STEP 1: Load & Inspect Data
# ---------------------------
print("\n==== STEP 1: LOAD & INSPECT DATA ====")
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}. Exiting.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded CSV: {CSV_PATH}")
print(f"Shape: {df.shape}")
print("Columns:", list(df.columns))
print("\nMissing values per important column (top):")
for c in ['temperature_recorded_date', 'apy_item_interval_start', 'state_name', 'state_temperature_max_val', 'state_temperature_min_val', 'state_rainfall_val', 'yield']:
    if c in df.columns:
        print(f"  {c}: {df[c].isna().sum()} missing")
print()

# normalize column names & strip spaces
df.columns = [c.strip() for c in df.columns]

# Choose date / year column
if 'temperature_recorded_date' in df.columns and df['temperature_recorded_date'].notna().any():
    print("Using 'temperature_recorded_date' as source of dates.")
    df['temperature_recorded_date'] = pd.to_datetime(
        df['temperature_recorded_date'], errors='coerce')
    df['year'] = df['temperature_recorded_date'].dt.year
else:
    print("Using 'apy_item_interval_start' as source of dates (parsed as year).")
    if 'apy_item_interval_start' not in df.columns:
        raise ValueError(
            "No date column found ('temperature_recorded_date' or 'apy_item_interval_start').")
    try:
        df['year'] = pd.to_datetime(
            df['apy_item_interval_start'], errors='coerce').dt.year
        if df['year'].isna().all():
            df['year'] = pd.to_numeric(
                df['apy_item_interval_start'], errors='coerce').astype('Int64')
    except Exception:
        df['year'] = pd.to_numeric(
            df['apy_item_interval_start'], errors='coerce').astype('Int64')

# drop rows without year
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

print(f"Year range: {int(df['year'].min())} — {int(df['year'].max())}")
if 'state_name' not in df.columns:
    raise ValueError("Input CSV must contain 'state_name' column.")
print("Unique raw state names (sample):", pd.Series(
    df['state_name'].dropna().unique()).head(30).tolist())

# ---------------------------
# Normalize state names to a canonical key
# ---------------------------


def make_state_key(name):
    if pd.isna(name):
        return "UNKNOWN"
    s = str(name).strip()
    # replace multiple spaces, remove punctuation, lowercase
    s = re.sub(r'\s+', ' ', s)
    s_clean = re.sub(r'[^0-9A-Za-z ]+', '', s).strip().lower()
    # make underscores for folder names
    s_key = re.sub(r'\s+', '_', s_clean)
    return s_key if s_key else "UNKNOWN"


df['state_raw'] = df['state_name'].astype(str)
df['state_key'] = df['state_raw'].apply(make_state_key)

# Create a mapping of state_key -> list of raw names (for audit)
state_map = df.groupby('state_key')['state_raw'].unique().to_dict()
# Save mapping
pd.DataFrame([(k, ",".join(v)) for k, v in state_map.items()], columns=['state_key',
             'raw_names']).to_csv(os.path.join(OUTPUT_ROOT, "state_key_mapping.csv"), index=False)

# Replace 'state_name' for grouping with the canonical 'state_key' but keep original raw names in outputs.
df['state_group'] = df['state_key']  # use for grouping/processing
print("Canonical state keys sample:", list(df['state_group'].unique())[:30])

# ---------------------------
# STEP 2: Reduce Data (yearly)
# ---------------------------
print("\n==== STEP 2: REDUCE DAILY → YEARLY AGGREGATION ====")
# Ensure numeric types exist
for col in ['state_temperature_max_val', 'state_temperature_min_val', 'state_rainfall_val', 'yield']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

agg_dict = {}
if 'state_temperature_max_val' in df.columns:
    agg_dict['state_temperature_max_val'] = 'mean'
if 'state_temperature_min_val' in df.columns:
    agg_dict['state_temperature_min_val'] = 'mean'
if 'state_rainfall_val' in df.columns:
    agg_dict['state_rainfall_val'] = 'sum'
if 'yield' in df.columns:
    agg_dict['yield'] = 'mean'

# group by canonical state_group and year
yearly = df.groupby(['state_group', 'year']).agg(
    agg_dict).reset_index().sort_values(['state_group', 'year'])
yearly_path = os.path.join(OUTPUT_ROOT, "yearly_aggregated_all_states.csv")
yearly.to_csv(yearly_path, index=False)
print(f"Saved yearly aggregated CSV: {yearly_path}")
print(f"Total rows in yearly aggregation: {len(yearly)}")

# ---------------------------
# STEP 3: Visualize combined & per-state (yield vs year)
# ---------------------------
print("\n==== STEP 3: VISUALIZATIONS (combined + per-state) ====")
states = yearly['state_group'].dropna().unique()

# combined yield plot - each canonical group plotted
plt.figure(figsize=(12, 6))
for st in states:
    s = yearly[yearly['state_group'] == st]
    plt.plot(s['year'], s['yield'], marker='o', label=st)
plt.title("Yield Over Time (All States - canonical groups)")
plt.xlabel("Year")
plt.ylabel("Yield")
plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
combined_yield_path = os.path.join(
    OUTPUT_ROOT, "all_states_yield_trend_canonical.png")
plt.savefig(combined_yield_path)
plt.close()
print(f"Saved combined yield plot (canonical): {combined_yield_path}")

# Combined rainfall & tmax if present
if 'state_rainfall_val' in yearly.columns:
    plt.figure(figsize=(12, 6))
    for st in states:
        s = yearly[yearly['state_group'] == st]
        plt.plot(s['year'], s['state_rainfall_val'], marker='o', label=st)
    plt.title("Rainfall Over Time (All States - canonical)")
    plt.xlabel("Year")
    plt.ylabel("Yearly Rainfall (sum)")
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    p = os.path.join(OUTPUT_ROOT, "all_states_rainfall_trend_canonical.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

if 'state_temperature_max_val' in yearly.columns:
    plt.figure(figsize=(12, 6))
    for st in states:
        s = yearly[yearly['state_group'] == st]
        plt.plot(s['year'], s['state_temperature_max_val'],
                 marker='o', label=st)
    plt.title("Max Temperature Over Time (All States - canonical)")
    plt.xlabel("Year")
    plt.ylabel("Tmax (mean)")
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    p = os.path.join(OUTPUT_ROOT, "all_states_tmax_trend_canonical.png")
    plt.savefig(p)
    plt.close()
    print(f"Saved: {p}")

# per-state csvs and plots (one canonical folder per state)
for st in states:
    s = yearly[yearly['state_group'] == st].sort_values('year')
    state_folder = os.path.join(OUTPUT_ROOT, st)
    os.makedirs(state_folder, exist_ok=True)
    # save CSV for canonical state (include raw names that mapped here)
    raw_names = state_map.get(st, [])
    out_csv = os.path.join(state_folder, f"{st}_yearly.csv")
    s_copy = s.copy()
    s_copy['raw_names'] = ",".join(raw_names)
    s_copy.to_csv(out_csv, index=False)
    # yield plot
    plt.figure(figsize=(8, 4))
    plt.plot(s['year'], s['yield'], marker='o', linestyle='-')
    plt.title(f"Yield vs Year - {st}")
    plt.xlabel("Year")
    plt.ylabel("Yield")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(state_folder, f"{st}_yield_trend.png")
    plt.savefig(path)
    plt.close()

print("Per-state (canonical) yield plots and yearly CSVs saved.")

# ---------------------------
# Helper: stationarity tests
# ---------------------------


def stationarity_tests(series):
    out = {}
    series_clean = series.dropna()
    if len(series_clean) < 3:
        return {'adf_p': None, 'kpss_p': None, 'adf_stat': None, 'kpss_stat': None}
    try:
        adf_res = adfuller(series_clean, autolag='AIC')
        out['adf_stat'] = adf_res[0]
        out['adf_p'] = adf_res[1]
    except Exception:
        out['adf_stat'] = None
        out['adf_p'] = None
    try:
        kpss_res = kpss(series_clean, regression='c', nlags='auto')
        out['kpss_stat'] = kpss_res[0]
        out['kpss_p'] = kpss_res[1]
    except Exception:
        out['kpss_stat'] = None
        out['kpss_p'] = None
    return out


# ---------------------------
# Steps 4–9: Per-state detailed pipeline (DECOMPOSE, TESTS, DIFF, MODEL, FORECAST)
# ---------------------------
print("\n==== STEPS 4–9: Per-state decomposition, tests, differencing, model, forecast ====")
summary_rows = []

for st in states:
    print(f"\n---- Processing canonical state: {st} ----")
    state_folder = os.path.join(OUTPUT_ROOT, st)
    os.makedirs(state_folder, exist_ok=True)

    # Read the per-state yearly DF we created
    s = yearly[yearly['state_group'] == st].sort_values(
        'year').reset_index(drop=True)

    # create summary file (start)
    summary_file = os.path.join(state_folder, f"{st}_analysis_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Canonical state key: {st}\n")
        f.write("Mapped raw names: " + ",".join(state_map.get(st, [])) + "\n\n")

    if s.empty:
        print(f"  No data for {st}, skipping.")
        with open(summary_file, "a") as f:
            f.write(
                "No data available for this state. Skipped all subsequent steps.\n")
        summary_rows.append({'state': st, 'model_trained': False})
        continue

    # Step 4: Decompose (trend)
    print("  STEP 4: Decomposition (additive & multiplicative if possible)")
    series = s['yield']

    try:
        if len(series.dropna()) >= 3:
            # Yearly data: use period=1 to get trend extraction as before
            add = seasonal_decompose(series.fillna(
                method='ffill'), model='additive', period=1, extrapolate_trend='freq')
            add.plot()
            plt.suptitle(f"Additive Decomposition - {st}")
            plt.tight_layout()
            plt.savefig(os.path.join(
                state_folder, f"{st}_additive_decomp.png"))
            plt.close()
            print("    Saved additive decomposition.")
        else:
            print("    Not enough data points for additive decomposition.")
            with open(summary_file, "a") as f:
                f.write("STEP 4: Not enough data for decomposition.\n")
    except Exception as e:
        print("    Additive decomposition failed:", e)
        with open(summary_file, "a") as f:
            f.write("STEP 4 decomposition error: " + str(e) + "\n")

    try:
        if (series.dropna() > 0).all() and len(series.dropna()) >= 3:
            mult = seasonal_decompose(series.fillna(
                method='ffill'), model='multiplicative', period=1, extrapolate_trend='freq')
            mult.plot()
            plt.suptitle(f"Multiplicative Decomposition - {st}")
            plt.tight_layout()
            plt.savefig(os.path.join(
                state_folder, f"{st}_multiplicative_decomp.png"))
            plt.close()
            print("    Saved multiplicative decomposition.")
        else:
            with open(summary_file, "a") as f:
                f.write(
                    "STEP 4: Multiplicative decomposition skipped (non-positive values or insufficient data).\n")
    except Exception as e:
        print("    Multiplicative decomposition failed:", e)
        with open(summary_file, "a") as f:
            f.write("STEP 4 multiplicative error: " + str(e) + "\n")

    # Step 5: Stationarity tests
    print("  STEP 5: Stationarity tests (ADF + KPSS)")
    st_tests = stationarity_tests(series)
    with open(summary_file, "a") as f:
        f.write("STEP 5: Stationarity tests results\n")
        f.write(
            f"  ADF stat: {st_tests.get('adf_stat')}, ADF p: {st_tests.get('adf_p')}\n")
        f.write(
            f"  KPSS stat: {st_tests.get('kpss_stat')}, KPSS p: {st_tests.get('kpss_p')}\n")

    print(
        f"    ADF p-value: {st_tests.get('adf_p')}, KPSS p-value: {st_tests.get('kpss_p')}")

    # Step 6: Differencing if required (always compute and save results)
    print("  STEP 6: Differencing if required")
    need_diff = False
    if st_tests.get('adf_p') is None:
        need_diff = False
    else:
        if st_tests['adf_p'] >= 0.05 or (st_tests.get('kpss_p') is not None and st_tests['kpss_p'] < 0.05):
            need_diff = True

    if need_diff:
        series_diff = series.diff().dropna()
        diff_tests = stationarity_tests(series_diff)
        with open(summary_file, "a") as f:
            f.write("\nSTEP 6: Differencing applied\n")
            f.write(
                f"  After diff ADF stat: {diff_tests.get('adf_stat')}, p: {diff_tests.get('adf_p')}\n")
            f.write(
                f"  After diff KPSS stat: {diff_tests.get('kpss_stat')}, p: {diff_tests.get('kpss_p')}\n")
        # save differenced series plot
        plt.figure(figsize=(8, 3))
        plt.plot(series_diff.index, series_diff.values, marker='o')
        plt.title(f"Differenced Yield - {st}")
        plt.xlabel("Index")
        plt.ylabel("Differenced Yield")
        plt.tight_layout()
        plt.savefig(os.path.join(state_folder, f"{st}_differenced_series.png"))
        plt.close()
        print("    Differenced series saved.")
    else:
        series_diff = series
        with open(summary_file, "a") as f:
            f.write(
                "\nSTEP 6: Differencing NOT required (series considered stationary enough)\n")
        print("    Differencing not required (logged).")

    # Step 7: Train-Test split (calendar-year)
    print("  STEP 7: Train-Test split (calendar-year based)")
    if MIN_TRAIN_YEAR is not None:
        s_filtered = s[s['year'] >= MIN_TRAIN_YEAR].reset_index(drop=True)
        if s_filtered.empty:
            with open(summary_file, "a") as f:
                f.write(
                    f"\nSTEP 7: No data after applying MIN_TRAIN_YEAR={MIN_TRAIN_YEAR}. Skipping modeling.\n")
            summary_rows.append({'state': st, 'model_trained': False})
            print(
                f"    No data after MIN_TRAIN_YEAR {MIN_TRAIN_YEAR}, skipping.")
            continue
        s = s_filtered

    first_year_available = int(s['year'].min())
    last_year_available = int(s['year'].max())
    train_end_year = last_year_available - TEST_YEARS
    train_start_year = first_year_available
    test_start_year = train_end_year + 1
    test_end_year = last_year_available

    with open(summary_file, "a") as f:
        f.write(
            f"\nSTEP 7: Data years available: {first_year_available} — {last_year_available}\n")
        f.write(
            f"Desired train years: {train_start_year} — {train_end_year}\n")
        f.write(f"Desired test years: {test_start_year} — {test_end_year}\n")

    train = s[(s['year'] >= train_start_year) & (
        s['year'] <= train_end_year)].set_index('year')
    test = s[(s['year'] >= test_start_year) & (
        s['year'] <= test_end_year)].set_index('year')

    if len(train) < MIN_POINTS_FOR_MODEL:
        print("    Not enough training points for reliable model. Skipping modeling for this state.")
        with open(summary_file, "a") as f:
            f.write("\nSTEP 7: Not enough training points. Skipping modeling.\n")
        summary_rows.append({'state': st, 'model_trained': False})
        continue

    with open(summary_file, "a") as f:
        f.write(
            f"STEP 7: Train years: {train.index.min()} - {train.index.max()} ({len(train)})\n")
        if len(test) > 0:
            f.write(
                f"STEP 7: Test years: {test.index.min()} - {test.index.max()} ({len(test)})\n")
        else:
            f.write("STEP 7: Test years: N/A (no test rows)\n")

    # Step 8: Fit ARIMAX (with exog if available)
    print("  STEP 8: Fit ARIMAX")
    exog_cols = [c for c in ['state_temperature_max_val',
                             'state_temperature_min_val', 'state_rainfall_val'] if c in train.columns]

    y_train = train['yield']

    try:
        # build model and fit
        if len(exog_cols) > 0:
            X_train = train[exog_cols]
            X_test = test[exog_cols] if len(test) > 0 else None
            model = SARIMAX(y_train, exog=X_train, order=(
                1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            model_type = "ARIMAX"
            print("    ARIMAX fitted with exogenous variables:", exog_cols)
        else:
            model = SARIMAX(y_train, order=(
                1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            model_type = "ARIMA"
            print("    ARIMA fitted (no exogenous variables available).")

        # save model summary
        with open(os.path.join(state_folder, f"{st}_model_summary.txt"), "w") as mf:
            mf.write(res.summary().as_text())
        print("    Model summary saved.")

        # Step 9: Forecast for specified years (2023,2024,2025)
        print("  STEP 9: Forecast for", FORECAST_YEARS)
        steps = len(FORECAST_YEARS)

        if len(exog_cols) > 0:
            # simple future exog: mean of last up-to-3 training rows (safe fallback)
            last_exog = X_train.tail(min(3, len(X_train))).mean().to_frame().T
            exog_future = pd.concat([last_exog] * steps, ignore_index=True)
            exog_future.index = FORECAST_YEARS
        else:
            exog_future = None

        pred = res.get_forecast(steps=steps, exog=exog_future)
        pred_mean = pred.predicted_mean
        conf_int = pred.conf_int()

        forecast_df = pd.DataFrame({
            'year': FORECAST_YEARS,
            'forecast': pred_mean.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        }).set_index('year')

        forecast_csv = os.path.join(
            state_folder, f"{st}forecast{FORECAST_YEARS[0]}_{FORECAST_YEARS[-1]}.csv")
        forecast_df.to_csv(forecast_csv)
        print(f"    Forecast saved: {forecast_csv}")

        # Save forecast plot (train + test + forecast)
        plt.figure(figsize=(8, 4))
        plt.plot(train.index, y_train, label='Train', marker='o')
        if len(test) > 0:
            plt.plot(test.index, test['yield'],
                     label='Test', marker='o', color='orange')

        plt.plot(FORECAST_YEARS, pred_mean.values,
                 label='Forecast', marker='o', color='red')
        plt.fill_between(
            FORECAST_YEARS, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        plt.title(f"{st} - Train/Test/Forecast")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        p = os.path.join(state_folder, f"{st}_train_test_forecast.png")
        plt.savefig(p)
        plt.close()
        print(f"    Forecast plot saved: {p}")

        # Evaluation on test set (if available)
        eval_mae = eval_rmse = None

        test_steps = len(test)
        if test_steps > 0:
            try:
                if len(exog_cols) > 0 and X_test is not None and len(X_test) == test_steps:
                    pred_test = res.get_forecast(steps=test_steps, exog=X_test)
                else:
                    pred_test = res.get_forecast(steps=test_steps)
                pred_test_mean = pred_test.predicted_mean

                if len(pred_test_mean) == len(test['yield']):
                    eval_mae = mean_absolute_error(
                        test['yield'].values, pred_test_mean.values)
                    eval_rmse = np.sqrt(mean_squared_error(
                        test['yield'].values, pred_test_mean.values))
                    print(
                        f"    Test MAE: {eval_mae:.4f}, RMSE: {eval_rmse:.4f}")
                    with open(summary_file, "a") as f:
                        f.write(
                            f"\nSTEP 9: Model evaluation on test\n  MAE: {eval_mae:.4f}\n  RMSE: {eval_rmse:.4f}\n")
                else:
                    print(
                        "    Test forecast length mismatch; skipping numeric evaluation on test.")
                    with open(summary_file, "a") as f:
                        f.write(
                            "\nSTEP 9: Test forecast length mismatch; evaluation skipped.\n")
            except Exception as e:
                print("    Could not compute evaluation on test set:", e)
                with open(summary_file, "a") as f:
                    f.write("\nSTEP 9 evaluation error: " + str(e) + "\n")
        else:
            print("    No test years to evaluate against.")
            with open(summary_file, "a") as f:
                f.write("\nSTEP 9: No test years available to evaluate against.\n")

        # write final status to summary
        with open(summary_file, "a") as f:
            f.write("\nFINAL: Model & Forecast\n")
            f.write(f"  Model type: {model_type}\n")
            f.write(f"  Exogenous columns used: {exog_cols}\n")
            f.write(f"  Forecast CSV: {os.path.basename(forecast_csv)}\n")
            if eval_mae is not None:
                f.write(f"  Test MAE: {eval_mae:.4f}, RMSE: {eval_rmse:.4f}\n")

        summary_rows.append({'state': st, 'model_trained': True, 'mae': eval_mae, 'rmse': eval_rmse,
                             'train_start': train.index.min(), 'train_end': train.index.max(),
                             'test_start': (test.index.min() if len(test) > 0 else None),
                             'test_end': (test.index.max() if len(test) > 0 else None)})
    except Exception as e:
        print(f"    Model fitting / forecasting failed for {st}: {e}")
        with open(summary_file, "a") as f:
            f.write("\nSTEP 8/9: Model failed with exception:\n")
            f.write(str(e) + "\n")
        summary_rows.append({'state': st, 'model_trained': False})
        continue

# Save overall model summary table
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(
    OUTPUT_ROOT, "all_states_model_summary.csv"), index=False)
print("\n==== PIPELINE COMPLETE ====")
print(f"All outputs saved under folder: {OUTPUT_ROOT}")
print("Top-level summary saved to:",
      os.path.join(OUTPUT_ROOT, "all_states_model_summary.csv"))
