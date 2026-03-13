
import json
import time
import re
import os
import shutil

import pandas as pd
import numpy as np
from google.cloud import bigquery
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import matplotlib.pyplot as plt
from tqdm import tqdm
from edgar import Company, set_identity

from scripts.config import ROOT_DIR, CACHE_DIR, MODEL_DIR, LEDGER_DIR, DATA_DIR

class EndError(Exception):
    pass

set_identity("Alex adlanecki@outlook.com")

##############################################################################################################

def temp():
    t = os.listdir(CACHE_DIR)

    for f in tqdm(t):
        try:
            parts = f.split("_")
            ticker = parts[0].upper()

            if not (os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_1d"))
                and os.path.exists(os.path.join(MODEL_DIR, f"{ticker}_1h"))):

                try: os.remove(os.path.join(LEDGER_DIR, f))
                except: pass

                try: shutil.rmtree(os.path.join(MODEL_DIR, f"{ticker}_1h"))
                except: pass
                try: shutil.rmtree(os.path.join(MODEL_DIR, f"{ticker}_1d"))
                except: pass

                try: os.remove(os.path.join(CACHE_DIR, f"{ticker}_1h.csv"))
                except: pass
                try: os.remove(os.path.join(CACHE_DIR, f"{ticker}_1d.csv"))
                except: pass

        except: pass

##############################################################################################################
""" sentiments """
# Download sentiment data
def main():
    sent_client = bigquery.Client(
        project="market-predictor-throwaway",
        client_options={"quota_project_id": "market-predictor-throwaway"}
    )

    print("Starting...")
    with open(os.path.join(ROOT_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)

    all_results = []
    company_names = list(ticker_map.keys())

    print("Creating regex...")
    half = len(company_names) // 2
    c1, c2 = company_names[:half], company_names[half:]

    r1 = "|".join([rf"\b{re.escape(name)}\b" for name in c1])
    r2 = "|".join([rf"\b{re.escape(name)}\b" for name in c2])

    for year in range(2016, 2027):
        try:
            print(f"Querying GDELT data for year {year}...")

            start_pt = f"{year}-01-01"
            end_pt = f"{year}-12-31"

            for i, reg_part in enumerate([r1, r2]):
                print()
                query = f"""
                        SELECT
                            DATE(_PARTITIONTIME) AS event_date,
                            LOWER(V2Organizations) AS organizations,
                            AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS avg_tone,
                            COUNT(*) AS article_count
                        FROM
                            -- Using the strictly partitioned table to save quota
                            `gdelt-bq.gdeltv2.gkg_partitioned`
                        WHERE
                            _PARTITIONTIME BETWEEN TIMESTAMP('{start_pt}') AND TIMESTAMP('{end_pt}')
                            AND REGEXP_CONTAINS(LOWER(V2Organizations), r'''({reg_part})''')
                        GROUP BY
                            event_date, organizations
                        HAVING 
                            article_count > 2
                        ORDER BY
                            event_date ASC
                    """

                try:
                    print(f"Querying GDELT data for part {i+1}...")
                    query_job = sent_client.query(query)
                    print(f"--> JOB ID: {query_job.job_id}")

                    # job_id = "PASTE_ID_HERE"
                    # query_job = client.get_job(job_id)

                    print("Converting to dataframe...")
                    start = time.perf_counter()
                    df = query_job.to_dataframe()
                    print(f"Completed in {time.perf_counter() - start:.2f} seconds")

                    if not df.empty:
                        print(f"Mapping {len(df)} rows to tickers...")

                        print("Finding tickers...")
                        df['matched'] = df['organizations'].str.extract(f'({reg_part})', flags=re.IGNORECASE, expand=False)
                        df['ticker'] = df['matched'].str.lower().map(ticker_map)

                        print("Appending to total...")
                        df = df.dropna(subset=['ticker'])
                        all_results.append(df[['ticker', 'event_date', 'avg_tone', 'article_count']])
                except KeyboardInterrupt:
                    raise EndError
                except Exception as e:
                    print(f"Failed on year {year} part {i+1}: {e}")

        except EndError:
            break

    if all_results:
        print("Writing results to file...")
        full_df = pd.concat(all_results, ignore_index=True)
        full_df.to_parquet("full_sent.parquet", index=False)
        print("Done!")
        print(full_df.head())
    else:
        print("No data retrieved.")

################################################
""" sentiments """
# Add sentiment technical indicators
def boost_features(df):
    print("Loading combined data...")
    df['event_date'] = pd.to_datetime(df['event_date'])

    # Collapse duplicate ticker/date entries into a single weighted average
    print("Aggregating duplicates (Weighted Average)...")
    df['weighted_tone'] = df['avg_tone'] * df['article_count']

    df = df.groupby(['ticker', 'event_date']).agg({
        'weighted_tone': 'sum',
        'article_count': 'sum'
    }).reset_index()

    # Recalculate the final daily tone
    df['avg_tone'] = df['weighted_tone'] / df['article_count']
    df = df.drop(columns=['weighted_tone'])

    # 1. Define US Market Days
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    market_days = pd.date_range(start='2015-02-18', end='2026-03-11', freq=us_bd)

    # 2. Reindexing - Creating a "Full Grid" (Ticker x Market Day)
    tickers = df['ticker'].unique()
    print(f"Expanding grid for {len(tickers)} tickers over {len(market_days)} market days...")

    # Create a MultiIndex of all Tickers and Market Days
    mux = pd.MultiIndex.from_product([tickers, market_days], names=['ticker', 'event_date'])

    # Set existing data to index and reindex to the full market grid
    df = df.set_index(['ticker', 'event_date']).reindex(mux).reset_index()

    # 3. Fill Gaps & Basic Columns
    print("Filling gaps and calculating basic features...")
    df['has_news'] = df['article_count'].notna().astype(int)
    df['article_count'] = df['article_count'].fillna(0)
    df['avg_tone'] = df['avg_tone'].fillna(0)

    # 4. Sentiment Impact: tone * log(count + 1)
    df['sentiment_impact'] = df['avg_tone'] * np.log1p(df['article_count'])

    # 5. Indicators (Grouped by Ticker)
    print("Calculating Indicators (this might take a minute)...")

    # We use transform to keep the same row count
    df['sma_7d'] = df.groupby('ticker')['avg_tone'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['sma_30d'] = df.groupby('ticker')['avg_tone'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    # Sentiment Volatility (Rolling Standard Deviation)
    df['sentiment_volatility_7d'] = df.groupby('ticker')['avg_tone'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
    )
    # Sentiment Momentum (The gap between short and long term vibes)
    # Positive = Sentiment is improving; Negative = Sentiment is cooling off
    df['sentiment_momentum'] = df['sma_7d'] - df['sma_30d']

    # Volume Z-Score (The "Shock" factor)
    # This identifies days when news volume is significantly higher than usual for THAT specific ticker
    df['volume_zscore'] = df.groupby('ticker')['article_count'].transform(
        lambda x: (x - x.rolling(30).mean()) / (x.rolling(30).std() + 1e-9)
    ).fillna(0)

    # 6. Final Save
    output_file = os.path.join(ROOT_DIR, "data", "sentiment_features.parquet")
    print(f"Saving model-ready data to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("Done! Ready for the model.")

# Remove them again
def revert_to_raw():
    path = os.path.join(ROOT_DIR, "data", "sentiment_features.parquet")
    df = pd.read_parquet(path)

    # List of engineered columns to remove
    to_drop = [
        'sentiment_impact', 'sma_7d', 'sma_30d',
        'sentiment_volatility_7d', 'sentiment_momentum', 'volume_zscore'
    ]

    # Drop only the columns that actually exist in the file
    df_raw = df.drop(columns=[c for c in to_drop if c in df.columns])

    # Save as your new "Source of Truth"
    new_path = os.path.join(ROOT_DIR, "data", "master_sentiment.parquet")
    df_raw.to_parquet(new_path, index=False)
    print(f"Reverted to raw data. Columns remaining: {df_raw.columns.tolist()}")

################################################
""" sentiments """
# Tests to ensure data integrity
def run_final_validation():
    input_file = os.path.join(ROOT_DIR, "data", "sentiment_features.parquet")
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)

    # 1. Statistical Outlier Detection
    print("\n--- Feature Statistics ---")
    cols_to_check = ['avg_tone', 'article_count', 'sentiment_impact', 'volume_zscore', 'sentiment_momentum']
    stats = df[cols_to_check].describe().transpose()
    print(stats[['min', 'max', 'mean', 'std']])

    # 2. Logic & Continuity Checks
    print("\n--- Integrity Checks ---")
    # Check for Infinite values
    inf_mask = np.isinf(df[cols_to_check]).any(axis=1)
    print(f"Rows with Infinite values: {inf_mask.sum()}")

    # Check for nulls
    null_counts = df.isnull().sum().sum()
    print(f"Total NULL values: {null_counts}")

    # Verify Market Days (No Weekends)
    weekends = df[df['event_date'].dt.dayofweek >= 5]
    print(f"Weekend rows found: {len(weekends)}")

    # 3. Visualizing a "High Intensity" Ticker
    # Picking NVDA as it was in your sample
    ticker_to_plot = 'NVDA'
    sample = df[df['ticker'] == ticker_to_plot].sort_values('event_date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Tone and SMAs
    ax1.plot(sample['event_date'], sample['avg_tone'], alpha=0.3, label='Raw Tone', color='gray')
    ax1.plot(sample['event_date'], sample['sma_7d'], label='7d SMA', color='blue')
    ax1.plot(sample['event_date'], sample['sma_30d'], label='30d SMA', color='red')
    ax1.set_title(f"Sentiment Trends: {ticker_to_plot}")
    ax1.legend()

    # Plot Volume Z-Score (The "Shock" factor)
    ax2.bar(sample['event_date'], sample['volume_zscore'], color='purple', alpha=0.6, label='Volume Shock')
    ax2.axhline(y=3, color='r', linestyle='--', label='High Volatility Threshold')
    ax2.set_title("News Volume Shocks (Z-Score)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Tests to ensure data integrity
def run_stress_test():
    input_file = os.path.join(ROOT_DIR, "data", "sentiment_features.parquet")
    df = pd.read_parquet(input_file)

    print("--- [STATISTICAL BOUNDARIES] ---")
    # Verify values are within expected ranges observed in your sample
    stats = df.describe().transpose()
    print(stats[['min', 'max', 'mean']])

    print("\n--- [LOGIC & ANOMALY CHECKS] ---")

    # 1. Check for Infinite/NaN values in indicators
    inf_count = np.isinf(df[['volume_zscore', 'sentiment_momentum']]).sum().sum()
    nan_count = df.isnull().sum().sum()
    print(f"Infinite values found: {inf_count}")
    print(f"Missing values (NaN) found: {nan_count}")

    # 2. Check the 'has_news' vs 'article_count' logic
    # Every row with article_count > 0 must have has_news = 1
    logic_fail = df[(df['article_count'] > 0) & (df['has_news'] == 0)]
    print(f"Logic Mismatches (article_count > 0 but has_news = 0): {len(logic_fail)}")

    # 3. Check for Extreme Volume Outliers
    # Z-scores above 10 are very rare; above 50 usually indicates a data error
    extreme_z = df[df['volume_zscore'] > 20]
    print(f"Extreme Volume Shocks (>20 sigma): {len(extreme_z)}")

    # 4. Weekend Check
    # Ensure CustomBusinessDay removed Saturdays and Sundays
    df['day_of_week'] = df['event_date'].dt.dayofweek
    weekend_rows = df[df['day_of_week'] >= 5]
    print(f"Weekend rows found (Should be 0): {len(weekend_rows)}")

    # 5. Visual Spot Check for a High-Volume Ticker
    print("\n--- [TIMELINE CONTINUITY: NVDA] ---")
    nvda = df[df['ticker'] == 'NVDA'].sort_values('event_date')
    print(nvda[['event_date', 'article_count', 'sma_7d', 'volume_zscore']].tail(5))

##############################################################################################################
""" post model checking """
def calculate_average_confidence():
    all_scores = []

    ledger_files = os.listdir(LEDGER_DIR)

    for file in ledger_files:
        df = pd.read_csv(os.path.join(LEDGER_DIR, file))

        for _, row in df.iterrows():
            prob = float(row['Probability'].replace("%", ""))
            direction = str(row['Direction'])

            if "DOWN" in direction:
                score = prob * -1 / 100
            else:
                score = prob / 100

            all_scores.append(score)

    if all_scores:
        avg_confidence = sum(all_scores) / len(all_scores)
        print(f"Processed {len(all_scores)} predictions.")
        print(f"Global Average Confidence Score: {avg_confidence:.4f}")

        if avg_confidence > 0:
            print("Overall Bias: BULLISH")
        else:
            print("Overall Bias: BEARISH")
    else:
        print("No valid prediction data found in ledgers.")

def remove_incomplete():
    ticker_list = {'FIGR', 'AMRZ', 'FNGU', 'TEM', 'VG', 'WM', 'GEV', 'RDDT', 'ALAB', 'KLAR', 'ETH', 'FIG', 'RAL', 'NBIS', 'GLXY', 'CRCL', 'VIK', 'Q', 'TTAN', 'RBRK', 'GRAL', 'SOLS', 'ETHA', 'SARO', 'KRMN', 'FETH', 'BMNR'}

    df = pd.read_parquet(os.path.join(DATA_DIR, "master_sentiment.parquet"))
    valid_tickers = json.load(open(os.path.join(ROOT_DIR, "valid_tickers.json")))
    valid_tickers_with_history = json.load(open(os.path.join(ROOT_DIR, "valid_tickers_with_history.json")))
    ticker_map = json.load(open(os.path.join(ROOT_DIR, "ticker_map.json")))

    for t in ticker_list:

        try: shutil.rmtree(os.path.join(MODEL_DIR, f"{t}_1h"))
        except: pass
        try: shutil.rmtree(os.path.join(MODEL_DIR, f"{t}_1d"))
        except: pass
        try: shutil.rmtree(os.path.join(CACHE_DIR, f"{t}"))
        except: pass

        try: os.remove(os.path.join(LEDGER_DIR, f"{t}_ledger.csv"))
        except: pass

        df = df[df['ticker'].str.upper() != t.upper()]

        try: valid_tickers.remove(t)
        except: pass
        try: valid_tickers_with_history.remove(t)
        except: pass

    ticker_map = {k: v for k, v in ticker_map.items() if v not in ticker_list}

    df.to_parquet(os.path.join(DATA_DIR, "master_sentiment.parquet"))

    with open(os.path.join(ROOT_DIR, "valid_tickers.json"), "w") as f:
        json.dump(valid_tickers, f)

    with open(os.path.join(ROOT_DIR, "valid_tickers_with_history.json"), "w") as f:
        json.dump(valid_tickers_with_history, f)

    with open(os.path.join(ROOT_DIR, "ticker_map.json"), "w") as f:
        json.dump(ticker_map, f)

def find_missing_files():
    caches = {f for f in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, f))}
    ledgers = {l.split("_")[0] for l in os.listdir(LEDGER_DIR)}
    models = {f.split("_")[0] for f in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, f))}
    sent_tickers = set(pd.read_parquet(os.path.join(DATA_DIR, "master_sentiment.parquet"))['ticker'].unique().tolist())

    # The '&' operator finds items present in ALL sets
    common = caches & ledgers & models
    # All unique tickers across every folder
    total = caches | ledgers | models
    # Tickers missing at least one component
    incomplete = total - common

    # Specific breakdowns
    only_cache = caches - (ledgers | models)
    missing_models = caches - models
    missing_ledgers = caches - ledgers
    not_sent = total - sent_tickers
    sent_not_other = sent_tickers - total

    ticker_map = json.load(open(os.path.join(ROOT_DIR, "ticker_map.json")))
    valid_tickers = json.load(open(os.path.join(ROOT_DIR, "valid_tickers.json")))
    valid_tickers_with_history = json.load(open(os.path.join(ROOT_DIR, "valid_tickers_with_history.json")))

    # 5. Print Results
    print(f"Ticker map: {len(ticker_map)}")
    print(f"Valid tickers: {len(valid_tickers)}")
    print(f"Valid tickers with history: {len(valid_tickers_with_history)}")
    print(f"Total Unique Tickers Found: {len(total)}")
    print(f"Complete Sets (Common): {len(common)}")
    print(f"Incomplete Sets: {len(incomplete)}")
    print(f"Sent Tickers: {len(sent_tickers)}")
    print(f"In sent not other: {len(not_sent)}")

    print(f"Missing Models: {missing_models}")
    print(f"Missing Ledgers: {missing_ledgers}")
    print(f"Only Cache: {missing_models}")
    print(f"Only Sent: {sent_not_other}")

    # with open(os.path.join(ROOT_DIR, "valid_tickers.json"), "w") as f:
    #     json.dump(list(total), f)
    #
    #
    # pattern = r',?\s+(inc|corp|ltd|co|llc|plc|l\.p\.|incorporated|corporation)\.?\s*$'
    # ticker_map = {}
    #
    # for symbol in tqdm(total):
    #     try:
    #         comp = Company(symbol)
    #         ticker_map[re.sub(pattern, '', comp.name, flags=re.IGNORECASE).strip()] = symbol
    #     except:
    #         ticker_map[symbol] = symbol
    #
    # with open(os.path.join(ROOT_DIR, "ticker_map.json"), "w") as f:
    #     json.dump(ticker_map, f)

##############################################################################################################

def cache_refactor():

    tickers = [f for f in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, f))]

    for ticker in tqdm(tickers):
        old_folder = os.path.join(CACHE_DIR, ticker)
        path_1h = os.path.join(old_folder, "1h_data.csv")
        path_1d = os.path.join(old_folder, "1d_data.csv")

        if os.path.exists(path_1h):
            shutil.move(path_1h, os.path.join(CACHE_DIR, f"{ticker}_1h.csv"))

        if os.path.exists(path_1d):
            shutil.move(path_1d, os.path.join(CACHE_DIR, f"{ticker}_1d.csv"))

        shutil.rmtree(old_folder)

def timefy():
    tickers = os.listdir(CACHE_DIR)

    for ticker in tqdm(tickers):
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, ticker), index_col=0, parse_dates=True)

            df.index = pd.to_datetime(df.index)
            # df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
            if ticker.split("_")[1] == "1h":
                df = df[:-7]
            else:
                df = df[:-2]

            df.index.name = "Date"

            df.to_csv(os.path.join(CACHE_DIR, ticker))

        except:
            df = df[:-7]

            df.index = pd.to_datetime(df.index)
            df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
            df.index.name = "Date"

            df.to_csv(os.path.join(CACHE_DIR, ticker))

##############################################################################################################

if __name__ in "__main__":
    # print("\n")
    # full_df = pd.read_parquet(os.path.join(SENT_DIR, "master_sentiment.parquet"))
    #
    # tickers = list(full_df['ticker'].unique())
    #
    # with open(os.path.join(ROOT_DIR, "ticker_history_list.json"), 'w') as f:
    #     json.dump(tickers, f)

    # revert_to_raw()

    calculate_average_confidence()

    # find_missing_files()
    # remove_incomplete()

    # import folder_trees
    # folder_trees.generate_tree(ROOT_DIR)

    # cache_refactor()

    # timefy()


    pass