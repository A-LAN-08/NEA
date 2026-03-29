
# from edgar import Company, set_identity
# set_identity("Name email@gmail.com")

##############################################################################################################
""" downloading data things """
def data_update():
    import os
    import pandas as pd
    import yfinance as yf
    from yfinance import shared
    from datetime import datetime, timezone
    from tqdm import tqdm

    from config import CACHE_DIR
    from data_management import NYSE_CAL, load_data

    with os.scandir(CACHE_DIR) as entries:
        files = [e for e in entries if e.is_file()]
        ticker_list = sorted(list({f.name.split("_")[0] for f in files}))

        # Find the entry with the oldest modification time
        oldest_file = min(files, key=lambda e: e.stat().st_mtime)
        start_date = os.path.getmtime(oldest_file.path) - pd.Timedelta(days=1)

    for interval in ["1h", "1d"]:
        # Download data
        shared._ERRORS = {}
        batch_data = yf.download(ticker_list, start=start_date, interval=interval,
                                 group_by='ticker', auto_adjust=False, progress=True)

        # If any failed, retry the download for just those
        if shared._ERRORS:
            print("Retrying failed tickers...")
            failed_tickers = list(shared._ERRORS.keys())
            shared._ERRORS = {}
            extra_data = yf.download(failed_tickers, start=start_date, interval=interval,
                                     group_by='ticker', auto_adjust=False, progress=True)

            if not extra_data.empty:
                batch_data = pd.concat([batch_data, extra_data], axis=1)

        # Find whether the market is open
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
        is_market_currently_open = False

        if not schedule.empty:
            mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
            mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
            is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

        for ticker in tqdm(ticker_list, desc=f"Processing {interval}"):
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
                updated_df = updated_df.loc[:, ~updated_df.columns.duplicated()]
                updated_df.to_csv(cache_path)

            except Exception: continue

def initial_download():
    import os
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timezone
    from tqdm import tqdm
    import json

    from config import CACHE_DIR, DATA_DIR
    from data_management import NYSE_CAL

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)

    ticker_list = sorted([f for f in ticker_map.values()])

    for interval in ["1h", "1d"]:
        for ticker in tqdm(ticker_list, desc=f"Downloading for {interval}"):
            try:
                ticker_df = yf.download(ticker, interval=interval, period="max", auto_adjust=False, progress=False)
                if ticker_df.empty: continue

                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = ticker_df.columns.get_level_values(0)

                now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
                schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)
                is_market_currently_open = False

                if not schedule.empty:
                    mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
                    mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)
                    is_market_currently_open = mkt_open <= now_utc_naive <= mkt_close

                if is_market_currently_open:
                    ticker_df = ticker_df.iloc[:-1]

                ticker_df.index = pd.to_datetime(ticker_df.index, utc=True).tz_localize(None)
                ticker_df.index.name = "Date"
                ticker_df.index = ticker_df.index.strftime('%Y-%m-%d %H:%M:%S')

                cache_path = os.path.join(CACHE_DIR, f"{ticker}_{interval}.csv")
                ticker_df.to_csv(cache_path)

            except Exception as e:
                tqdm.write(f"Error updating {ticker}: {e}")
                continue

def get_spy():
    import pandas as pd
    import os
    from config import DATA_DIR
    import yfinance as yf
    from datetime import datetime, timezone
    from data_management import NYSE_CAL

    for interval in ["1h", "1d"]:
        # df = pd.read_csv(os.path.join(DATA_DIR, f"SPY_{interval}.csv"))

        data = yf.download("SPY", interval=interval, period="max", auto_adjust=False, progress=False)

        # Flattens columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            cols: pd.MultiIndex = data.columns
            data.columns = cols.get_level_values(0)

        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
        data.index.name = "Date"

        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)

        if not schedule.empty:
            mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
            mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)

            # If we are currently between open and close, the last downloaded row is "Live"
            if mkt_open <= now_utc_naive <= mkt_close:
                data = data.iloc[:-1]

        data.to_parquet(os.path.join(DATA_DIR, f"SPY_{interval}.parquet"))

def get_special(key):
    import pandas as pd
    import os
    from config import DATA_DIR
    import yfinance as yf
    from datetime import datetime, timezone
    from data_management import NYSE_CAL

    for interval in ["1h", "1d"]:
        data = yf.download(key, interval=interval, period="max", auto_adjust=False, progress=True)

        if data.empty:
            print("Empty data")
            continue

        # Flattens columns if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            cols: pd.MultiIndex = data.columns
            data.columns = cols.get_level_values(0)

        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
        data.index.name = "Date"

        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        schedule = NYSE_CAL.schedule(start_date=now_utc_naive, end_date=now_utc_naive)

        if not schedule.empty:
            mkt_open = schedule.iloc[0]['market_open'].replace(tzinfo=None)
            mkt_close = schedule.iloc[0]['market_close'].replace(tzinfo=None)

            # If we are currently between open and close, the last downloaded row is "Live"
            if mkt_open <= now_utc_naive <= mkt_close:
                data = data.iloc[:-1]

        data.to_parquet(os.path.join(DATA_DIR, f"{key}_{interval}.parquet"))

##############################################################################################################
"""testing pipeline"""
def test_train():
    import pandas as pd
    import os

    from scripts.config import MODEL_DIR
    from scripts.predictor import TrainingManager, all_ticker_models_exist, Settings
    from scripts.data_management import load_data

    ticker = "ASML"
    interval = "1h"

    Settings.LOGGING = True

    full_data = load_data(ticker, interval)
    cutoff_date = full_data.index.max() - pd.Timedelta(days=(60 if interval == "1d" else 20))
    training_data = full_data[full_data.index < cutoff_date]

    if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_{interval}"), interval):
        trainer = TrainingManager()
        trainer.run_training_pipeline(ticker, interval, override_data=training_data)

def test_predict():
    import os
    import json
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from lightgbm import Booster as LGBMBooster
    from safetensors.torch import load_file

    from scripts.config import MODEL_DIR, DATA_DIR
    from scripts.data_management import load_data
    from scripts.predictor import LSTMBrain, save_prediction, get_market_dates
    import scripts.indicators # noqa

    ##### SETUP

    ticker = "AAPL"
    interval = "1d"

    full_data = load_data(ticker, interval)
    model_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")

    with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "model_hyperparameters.json"), 'r') as f:
        hyper_meta = json.load(f)

    ##### AI BRAINS

    scaler = joblib.load(f"{model_folder}/scaler.joblib")
    features = joblib.load(f"{model_folder}/features.joblib")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_registry = {}
    horizons = {1:1, 2:2, 4:4, 8:25} if "h" in interval else {1:1, 2:2, 5:7, 21:28}
    period = "h" if interval == "1h" else "d"

    for step in horizons.keys():
        horizon_folder = os.path.join(model_folder, f"{step}_horizon_models")
        if not os.path.exists(horizon_folder): continue

        model_registry[step] = {"models": {}, "weights": {}}
        global_meta = hyper_meta.get(f"{step}{period}", [])

        for model_filename in os.listdir(horizon_folder):
            model_path = os.path.join(horizon_folder, model_filename)

            # Identify model type
            if ".safetensors" in model_filename:
                model_type = "LSTM"
                params = next(m["best_params"] for m in global_meta if m["model_type"] == model_type)
                # Initialize brain once
                brain = LSTMBrain(
                    input_dim=len(features),
                    hidden_dim=params["module__hidden_dim"],
                    layers=params["module__layers"],
                    dropout=params["module__dropout"],
                )
                brain.load_state_dict(load_file(model_path))
                brain.to(device).eval()
                model_registry[step]["models"][model_type] = brain

            elif ".txt" in model_filename:
                model_type = "LGBM"
                model_registry[step]["models"][model_type] = LGBMBooster(model_file=model_path)

            else:
                model_type = "SVC" if "SVC" in model_filename else "Lasso"
                model_registry[step]["models"][model_type] = joblib.load(model_path)

            mcc_val = next((m["mcc"] for m in global_meta if m["model_type"] == model_type), 0)
            if mcc_val > 0:
                ticker_weight = meta.get(f"{model_type}_result", {}).get("absolute_sharpe", 0)
                global_weight = next((abs(m["sharpe_ratio"]) for m in global_meta if m["model_type"] == model_type), 0)
                model_registry[step]["weights"][model_type] = (ticker_weight * 0.6) + (global_weight * 0.4)
            else:
                model_registry[step]["weights"][model_type] = 0

    ##### Walk forward predictions

    last_train_date = pd.to_datetime(meta["training data end"])
    test_data = full_data[full_data.index > last_train_date]

    history = full_data[full_data.index <= last_train_date].tail(400).copy()
    processed_df = history.ind.add_indicators(ticker, interval)

    for current_time in test_data.index:
        current_price = processed_df['Adj Close'].iloc[-1]
        current_volatility_atr = float(processed_df['ATR'].iloc[-1])

        target_dates = get_market_dates(current_time, horizons, period)
        if len(target_dates) < 1: continue

        step_forecasts = {}
        for step, bundle in model_registry.items():
            if target_dates[step] is None: continue

            probs = {}
            for model_type, model_obj in bundle["models"].items():
                if model_type == "LSTM":
                    recent_data = processed_df[features].tail(14)
                    scaled_seq = scaler.transform(recent_data)
                    x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)

                    with torch.no_grad():
                        probs[model_type] = float(model_obj(torch.from_numpy(x_3d).to(device)).item())

                elif model_type == "LGBM":
                    scaled_row = scaler.transform(processed_df[features].iloc[-1:])
                    probs[model_type] = float(model_obj.predict(scaled_row)[0])

                else:  # Lasso / SVC
                    scaled_row = scaler.transform(processed_df[features].iloc[-1:])
                    probs[model_type] = float(model_obj.predict_proba(scaled_row)[0][1])

            weights = bundle["weights"]
            total_weight = sum(weights.values())
            avg_up_proba = sum(probs[m] * weights[m] for m in probs) / total_weight if total_weight > 0 else 0.5

            # Calculate whether it will go up or down
            adjusted_probability = max(avg_up_proba, 1 - avg_up_proba)
            direction = "UP ▲" if avg_up_proba > 0.5 else "DOWN ▼"

            # Calculate predicted price
            direction_multiplier = 1 if avg_up_proba > 0.5 else -1
            confidence_strength = 2 * (adjusted_probability - 0.5)
            expected_move_magnitude = current_volatility_atr * np.sqrt(step)

            predicted_price = current_price + (direction_multiplier * expected_move_magnitude * confidence_strength)
            capped_width = min(expected_move_magnitude * (1.0 + confidence_strength), current_price * 0.15)

            step_forecasts[step] = {
                "current_price": current_price,
                'price': predicted_price,
                'up': predicted_price + capped_width,
                'lo': predicted_price - capped_width,
                'target_date': target_dates[step],
                'time_difference': horizons[step],
                'avg_probability': adjusted_probability,
                'dir': direction,
                'LSTM_probability': probs["LSTM"],
                'LGBM_probability': probs["LGBM"],
                'SVC_probability': probs["SVC"],
                'LASSO_probability': probs["Lasso"],
            }

        save_prediction(ticker, interval, current_time, step_forecasts)

##############################################################################################################

if __name__ in "__main__":

    # from predictor import run_prediction_pipeline
    # print("Starting...")
    # r = run_prediction_pipeline("AAPL", "1h")
    # print(r)

    # initial_download()

    # import folder_trees
    # folder_trees.generate_tree("C:/Users/adlan_3zfnjq7/Desktop/Alex - Main/Projects/Stock Market Predictor/models/LH_1h")

    # find_missing_files()

    # get_special("^VIX")
    # get_special("^VVIX")
    # get_special("^TYX")

    test_train()
    # test_predict()

    pass