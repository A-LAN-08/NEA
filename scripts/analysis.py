
import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from scripts.config import CACHE_DIR, ROOT_DIR
from predictor import run_prediction_pipline
from data_management import UpdateWorker, load_data, NYSE_CAL

############################################################################

def process_single_ticker(ticker):
    for interval in ["1h", "1d"]:
        try:
            run_prediction_pipline(ticker, interval)
        except Exception as e:
            pass

def run_batch_predictions():
    with open(os.path.join(ROOT_DIR, "valid_tickers.json"), "r") as f:
        ticker_list = json.load(f)

    if not ticker_list:
        print("No tickers found in valid_tickers_with_history.json")
        return

    print("--- Updating Global Sentiment ---")
    updater = UpdateWorker()
    updater.sentiment_update()

    print(f"\n--- Batch Updating Prices for {len(ticker_list)} tickers ---")

    for interval in ["1h", "1d"]:
        first_ticker = ticker_list[0]
        first_df = load_data(first_ticker, interval)
        start_date = first_df.index[-1] - pd.Timedelta(days=1)

        print(f"Fetching {interval} data since {start_date}...")

        batch_data = yf.download(
            ticker_list,
            start=start_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            progress=True
        )

        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
        is_market_currently_open = False

        if not schedule.empty:
            mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
            mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
            is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

        print()
        for ticker in tqdm(ticker_list, desc=f"Processing {interval}"):
            try:
                new_rows = batch_data[ticker].dropna(how='all')

                if new_rows.empty: continue

                new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None)
                if is_market_currently_open:
                    new_rows = new_rows.iloc[:-1]

                cache_path = os.path.join(CACHE_DIR, ticker, f"{interval}_data.csv")
                existing_df = load_data(ticker, interval)

                updated_df = pd.concat([existing_df, new_rows])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df.to_csv(cache_path)

            except Exception as e:
                print(f"Error updating {ticker}: {e}")

    print("\n--- Running Predictions ---")

    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"-> Using {num_cores} CPU cores...")

    Parallel(n_jobs=num_cores)(
        delayed(process_single_ticker)(ticker) for ticker in tqdm(ticker_list, desc="Predicting...")
    )

############################################################################

if __name__ == '__main__':
    # target_time = datetime.datetime(2026, 3, 12, 14, 0, 0, tzinfo=datetime.timezone.utc)
    #
    # with time_machine.travel(target_time):
    #     run()

    run_batch_predictions()







