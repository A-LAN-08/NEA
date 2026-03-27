
import os
import json
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import logging
import pandas as pd
import joblib
import numpy as np
import torch
from lightgbm import Booster as LGBMBooster
from safetensors.torch import load_file

from scripts.config import DATA_DIR, MODEL_DIR
from scripts.predictor import TrainingManager, Settings, all_ticker_models_exist, LSTMBrain, save_prediction, get_market_dates
from scripts.data_management import load_data, UpdateWorker
import scripts.indicators  # noqa

logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)

############################################################################

def train_model(ticker):
    for interval in ["1h", "1d"]:
        full_data = load_data(ticker, interval)
        cutoff_date = full_data.index.max() - pd.Timedelta(days=(60 if interval == "1d" else 20))
        training_data = full_data[full_data.index < cutoff_date]

        if not all_ticker_models_exist(os.path.join(MODEL_DIR, f"{ticker}_{interval}"), interval):
            trainer = TrainingManager()
            trainer.run_training_pipeline(ticker, interval, override_data=training_data)

def run_training(free_cores):
    print("\n--- Training Models ---")
    num_cores = max(1, multiprocessing.cpu_count() - free_cores)
    print(f"-> Using {num_cores} CPU cores...")

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)
        ticker_list = sorted(list(ticker_map.values()))#[::-1]

    Parallel(n_jobs=num_cores)(
        delayed(train_model)(ticker) for ticker in tqdm(ticker_list, desc="Training")
    )

############################################################################

def predict_model(ticker, interval):
    ##### SETUP
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

def run_predictions(free_cores):
    print("\n--- Predicting Stocks ---")
    num_cores = max(1, multiprocessing.cpu_count() - free_cores)
    print(f"-> Using {num_cores} CPU cores...")

    with open(os.path.join(DATA_DIR, "ticker_map.json"), "r") as f:
        ticker_map = json.load(f)
        ticker_list = sorted(list(ticker_map.values()))  # [::-1]

    Parallel(n_jobs=num_cores)(
        delayed(predict_model)(ticker) for ticker in tqdm(ticker_list, desc="Predicting")
    )

############################################################################

def updates(sent, spy, cache):
    updater = UpdateWorker()
    if sent:
        print("--- Updating Global Sentiment ---")
        updater.sentiment_update()
    if spy:
        print("--- Updating Global Comparison data ---")
        updater.update_comparatives()
    if cache:
        print(f"--- Updating Prices for tickers ---")
        updater.data_updater()


if __name__ == '__main__':
    # import time_machine
    # target_time = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
    # with time_machine.travel(target_time):
        # run_batch_predictions()

    updates(  # Whether to update:
        sent=False,  # News sentiment
        spy=False,  # Market sentiment indicators
        cache=False,  # Stock cache
    )

    Settings.VERBOSE = 0 # Change to 0 if you don't want logging clogging up console
    run_training(
        free_cores=4 # How many CPU cores do you want left free
    )

    run_predictions(
        free_cores=4 # How many CPU cores do you want left free
    )







