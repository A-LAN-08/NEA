
import os
import json
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

from scripts.config import DATA_DIR
from scripts.predictor import run_prediction_pipeline, Settings
from scripts.data_management import UpdateWorker

############################################################################

def process_single_ticker(ticker):
    for interval in ["1h", "1d"]:
        try:
            run_prediction_pipeline(ticker, interval)
        except Exception as e:
            pass

def run_batch_predictions(sent, spy, cache, models, free_cores):
    if any([sent, spy, cache]):
        updater = UpdateWorker()
        if sent:
            print("--- Updating Global Sentiment ---")
            updater.sentiment_update()
        if spy:
            print("--- Updating Global SPY data ---")
            updater.update_spy()
        if cache:
            print(f"--- Updating Prices for tickers ---")
            updater.data_updater()

    if models:
        print("\n--- Running Predictions ---")
        num_cores = max(1, multiprocessing.cpu_count() - free_cores)
        print(f"-> Using {num_cores} CPU cores...")

        with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
            ticker_map = json.load(f)
            ticker_list = list(ticker_map.keys())

        Parallel(n_jobs=num_cores)(
            delayed(process_single_ticker)(ticker) for ticker in tqdm(ticker_list, desc="Predicting")
        )

############################################################################

if __name__ == '__main__':
    # import time_machine
    # target_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
    # with time_machine.travel(target_time):
        # run_batch_predictions()

    Settings.VERBOSE = 1 # Change to 0 if you don't want logging clogging up console

    # initial_download() # Run this if no stock cache downloaded yet

    run_batch_predictions(
        sent=True, spy=True, cache=True, # Whether to update: News sentiment, SPY data (market sentiment indicator), stock cache
        models=True, # Whether to train new models
        free_cores=2 # How many CPU cores do you want left free
    )







