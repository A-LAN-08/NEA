
import os
import json
import logging
import pandas as pd
import yfinance as yf
from yfinance import shared
from datetime import datetime, timezone
import tqdm
import multiprocessing
from joblib import Parallel, delayed
import time_machine

from scripts.config import CACHE_DIR, DATA_DIR
from predictor import run_prediction_pipeline
from data_management import UpdateWorker, load_data, NYSE_CAL

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

############################################################################

def process_single_ticker(ticker):
    for interval in ["1h", "1d"]:
        try:
            run_prediction_pipeline(ticker, interval)
        except Exception as e:
            pass

def run_batch_predictions():
    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)

    print("--- Updating Global Sentiment ---")
    updater = UpdateWorker()
    updater.sentiment_update()

    print(f"\n--- Batch Updating Prices for {len(ticker_map)} tickers ---")
    ticker_list = [f for f in ticker_map.values()]

    for interval in ["1h", "1d"]:
        first_ticker = ticker_list[0]
        first_df = load_data(first_ticker, interval)
        start_date = first_df.index[-1] - pd.Timedelta(days=8)

        print(f"Fetching {interval} data since {start_date}...")

        shared._ERRORS = {}

        batch_data = yf.download(
            ticker_list,
            start=start_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            progress=True
        )

        errors = shared._ERRORS
        shared._ERRORS = {}
        if errors:
            print("Retrying failed tickers...")
            failed_tickers = list(errors.keys())
            extra_data = yf.download(
                failed_tickers,
                start=start_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=False,
                progress=True
            )

            if not extra_data.empty:
                batch_data = pd.concat([batch_data, extra_data], axis=1)

        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
        is_market_currently_open = False

        if not schedule.empty:
            mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
            mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
            is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

        print()
        for ticker in tqdm.tqdm(ticker_list, desc=f"Processing {interval}"):
            try:
                new_rows = batch_data[ticker].dropna(how='all')

                if new_rows.empty: continue

                new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None)
                if is_market_currently_open:
                    new_rows = new_rows.iloc[:-1]

                new_rows.index.name = "Date"
                new_rows.index = pd.to_datetime(new_rows.index, utc=True).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')

                cache_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
                existing_df = load_data(ticker, interval)

                updated_df = pd.concat([existing_df, new_rows])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df.to_csv(cache_path)

            except Exception as e:
                tqdm.write(f"Error updating {ticker}: {e}")
                continue

    print("\n--- Running Predictions ---")

    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"-> Using {num_cores} CPU cores...")

    Parallel(n_jobs=num_cores)(
        delayed(process_single_ticker)(ticker) for ticker in tqdm.tqdm(ticker_list, desc="Predicting")
    )

############################################################################

if __name__ == '__main__':
    target_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)

    with time_machine.travel(target_time):
        # run()

        run_batch_predictions()







