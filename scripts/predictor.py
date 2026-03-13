
# Standard library imports
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta, time

# External library imports
import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import talib
import yfinance as yf
from lightgbm import LGBMClassifier
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Set environment variables and filters
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from data_management import load_data
from scripts.config import LEDGER_DIR, MODEL_DIR, DATA_DIR

############################################################################

# Class to create separate thread for trainer so can run concurrently with gui
class TrainingWorker(QThread):
    # Signal to send the results back to the GUI when finished
    training_finished: pyqtSignal = pyqtSignal(dict)
    training_error: pyqtSignal = pyqtSignal(str)

    def __init__(self, ticker: str, interval: str):
        super().__init__()
        self.ticker = ticker
        self.interval = interval

    # Run the prediction for its instance
    def run(self):
        try:
            forecast_results = run_prediction_pipline(self.ticker, self.interval)
            self.training_finished.emit(forecast_results)
        except Exception as e:
            self.training_error.emit(str(e))

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 42
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    @staticmethod
    def _integrate_sentiment(df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        target_file = os.path.join(DATA_DIR, f"master_sentiment.parquet")

        sent_df = pd.read_parquet(target_file, filters=[('ticker', '==', ticker)])

        sent_df['event_date'] = pd.to_datetime(sent_df['event_date'])
        sent_df = sent_df.sort_values('event_date')

        # Sentiment Impact: tone * log(count + 1)
        sent_df['sentiment_impact'] = sent_df['avg_tone'] * np.log1p(sent_df['article_count'])

        # Sentiment simple moving averages
        sent_df['sentiment_sma_7d'] = sent_df['avg_tone'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        sent_df['sentiment_sma_30d'] = sent_df['avg_tone'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

        # Sentiment Volatility (Rolling Standard Deviation)
        sent_df['sentiment_volatility_7d'] = sent_df['avg_tone'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
        )
        # Sentiment Momentum (The gap between short and long term vibes)
        # Positive = Sentiment is improving; Negative = Sentiment is cooling off
        sent_df['sentiment_momentum'] = sent_df['sentiment_sma_7d'] - sent_df['sentiment_sma_30d']

        # Sentiment volume Z-Score (The "Shock" factor)
        # This identifies days when news volume is significantly higher than usual for THAT specific ticker
        sent_df['sentiment_volume_zscore'] = sent_df['article_count'].transform(
            lambda x: (x - x.rolling(30).mean()) / (x.rolling(30).std() + 1e-9)
        ).fillna(0)

        sent_df['merge_date'] = sent_df['event_date'] + pd.Timedelta(days=1)
        sent_df = sent_df[sent_df.columns.difference(['ticker', 'event_date'])]

        index_name = df.index.name if df.index.name else 'index'
        df['merge_date'] = pd.to_datetime(df.index).normalize()
        df = df.reset_index()

        df = pd.merge(df, sent_df, on='merge_date', how='left')
        df = df.set_index(index_name)
        df = df.drop(columns=['merge_date']).fillna(0)

        return df

    # Calculate technical indicators
    def calculate_technical_indicators(self, df: pd.DataFrame, ticker: str, interval: str, training: bool = True) -> pd.DataFrame:
        df = self._integrate_sentiment(df, ticker, interval)

        # Horizon (h) targets for selected period (p)
        p = "h" if "h" in interval else "d"
        for h in [1, 2, 4, 5, 8, 21]:
            df[f'target_cls_{h}{p}'] = df['Close'].shift(-h).gt(df['Close']).astype(int)

        df['return'] = df['Close'].pct_change()
        for i in range(1, 4): df[f'return_lag_{i}'] = df['return'].shift(i)

        # Technical indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, _, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Hist'] = macdhist
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['MA_200'] = talib.SMA(df['Close'], timeperiod=200)
        df['PDMA_200'] = (df['Close'] / df['MA_200']) - 1

        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        upper, mid, lower = talib.BBANDS(df['Close'], timeperiod=20)
        df['BBP'] = (df['Close'] - lower) / (upper - lower)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

        # Deviations using all of OHLC
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['body_pct'] = (df['Close'] - df['Open']) / df['Close']
        df['upper_shadow_pct'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['lower_shadow_pct'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # Other indicators
        df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        if not training:
            df.loc[df['hour'] < 9, 'hour'] = 9

        if training:
            df = df.dropna()

        return df

    # Evaluate model performance with accuracy and sharpe ratio
    @staticmethod
    def _evaluate_performance(actual_direction: pd.Series, predicted_direction: np.ndarray, actual_returns: np.ndarray):
        # How often was the AI right about Up vs Down?
        hit_rate = accuracy_score(actual_direction, predicted_direction)
        # Following the AI, what would a daily wallet look like?
        daily_strategy_returns = actual_returns * predicted_direction
        # How risky were those returns
        volatility = daily_strategy_returns.std()
        # Calculate the reward-to-risk ratio (Annualized)
        sharpe_ratio = (daily_strategy_returns.mean() / volatility) * np.sqrt(252) if volatility != 0 else 0

        # Return the score
        return hit_rate, sharpe_ratio, abs(sharpe_ratio), (sharpe_ratio < 0)

    # Train and evaluate LightGBM with walk-forward validation
    def _train_lightgbm(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                        target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) so it matches inference
        data_normalizer = StandardScaler()
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Handle class imbalance (Make 'Up' and 'Down' days equally important)
        scale_pos_weight = (len(target_train) - target_train.sum()) / target_train.sum()

        # Initialize the LightGBM model
        model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=self.seed,
                               scale_pos_weight=scale_pos_weight, verbose=-1,
                               device="gpu", gpu_platform_id=0, gpu_device_id=0)

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_index, val_index in time_splitter.split(features_train):
            # Train on the "past" segment, test on the "future" segment
            model.fit(features_train_normalized.iloc[train_index], target_train.iloc[train_index])
            validation_scores.append(model.score(features_train_normalized.iloc[val_index], target_train.iloc[val_index]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions,price_returns)
        # Return all the information from the model
        return {'model_type': 'LGBM', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
                'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
                'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Train and evaluate Lasso Logistic Regression with walk-forward validation
    def _train_lasso_regression(self, features_train: pd.DataFrame, target_train: pd.Series,
                                features_test: pd.DataFrame, target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range)
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Logistic Regression with a "Lasso" (L1) penalty
        model = LogisticRegression(
            penalty='l1',  # Kills off useless features
            solver='liblinear',  # Required math solver for L1
            random_state=self.seed,
            class_weight='balanced'  # Automatically handles Up/Down day imbalance
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {'model_type': 'Lasso', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Train and evaluate SVC with walk-forward validation
    def _train_support_vector(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                              target_test: pd.Series, price_returns: np.ndarray) -> dict:
        # Put all indicators on the same scale (0 to 1 range) [SVC very sensitive to scale]
        data_normalizer = StandardScaler()

        # Scale the training and test data while keeping the column names
        features_train_normalized = pd.DataFrame(data_normalizer.fit_transform(features_train), columns=features_train.columns)
        features_test_normalized = data_normalizer.transform(features_test)

        # Initialize Support Vector model
        model = SVC(
            kernel='rbf',  # Allows for curved boundaries
            C=1.0,  # Penalty - how much weight given to misclassified days
            random_state=self.seed,
            class_weight='balanced', # Automatically handles Up/Down day imbalance
            probability=True  # To get probability scores
        )

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []
        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            model.fit(features_train_normalized.iloc[train_idx], target_train.iloc[train_idx])
            validation_scores.append(model.score(features_train_normalized.iloc[val_idx], target_train.iloc[val_idx]))

        # Final test on totally unseen data
        model.fit(features_train_normalized, target_train)
        test_predictions = model.predict(features_test_normalized)

        # Get the score
        accuracy, sharpe, abs_sharpe, needs_flip = self._evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {
            'model_type': 'SVC', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    # Save models for all horizons with the best performing model type
    def _save_winning_strategy_assets(self, ticker: str, interval: str, full_results: dict,
                                      features_data: dict, targets_dataframe: pd.DataFrame) -> None:
        # Create the folder for this specific stock's model data to save
        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        model_bundle = {
            "classifiers": {},
            "horizons": [1, 2, 4, 5, 8, 21]
        }

        best_model_type = max(full_results, key=lambda result: result['absolute_sharpe'])['model_type']
        period = "h" if "h" in interval else "d"

        scaler = StandardScaler().fit(features_data)
        scaled_features = scaler.transform(features_data)

        meta = {
            "results": [{k: (v.item() if hasattr(v, 'item') else v) for k, v in d.items() if k not in {"raw_predictions", "trained_model_object", "feature_scaler"} }
                    for d in full_results],
            "training date": datetime.now().strftime("%Y-%m-%d"),
            "best model": best_model_type
        }

        for h in model_bundle["horizons"]:
            target_col = f'target_cls_{h}{period}'
            # We must dropna for this specific target just to train this specific horizon
            valid_idx = targets_dataframe[target_col].notna()

            # Train Classifier
            if best_model_type == 'LGBM':
                model = LGBMClassifier(n_estimators=100, random_state=self.seed, verbose=-1)
            elif best_model_type == 'Lasso':
                model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
            else:
                model = SVC(kernel='rbf', probability=True, class_weight='balanced')

            model.fit(scaled_features[valid_idx], targets_dataframe[target_col][valid_idx])
            model_bundle["classifiers"][h] = model

        with open(f'{save_folder}/metadata.json', 'w') as f: json.dump(meta, f)
        joblib.dump(model_bundle, f"{save_folder}/models.pkl")
        joblib.dump(scaler, f"{save_folder}/scaler.pkl")
        joblib.dump(list(features_data.columns), f"{save_folder}/features.pkl")

    # Run all helper functions and consolidate the best model
    def run_training_pipeline(self, ticker: str, interval: str) -> bool:
        # Remove any corrupt model paths (i.e. not all necessary files exist validated before call)
        model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if os.path.exists(model_path): shutil.rmtree(model_path)

        # Train and build models for ticker
        # Load data and add indicators
        data = load_data(ticker, interval)
        if data is None: print("No data"); return False

        df = self.calculate_technical_indicators(data, ticker, interval)
        if len(df) < 300: print(f"Insufficient data for {ticker} (need 300+, got {len(df)})"); return False

        # Partition data
        train_size = int(len(df) * (1 - self.__test_size))
        train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

        # Remove "Cheat" columns and raw price data AI shouldn't see directly
        # ('target' as is answer key, 'Close' as is too easy to cheat with)
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        period = "h" if 'h' in interval else "d"
        features_train, features_test = train_data[train_columns], test_data[train_columns]
        targets_train, targets_test = train_data[f'target_cls_1{period}'], test_data[f'target_cls_1{period}']
        actual_returns_test = test_data['return'].values

        # Model competition
        results = [
            self._train_lightgbm(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_lasso_regression(features_train, targets_train, features_test, targets_test, actual_returns_test),
            self._train_support_vector(features_train, targets_train, features_test, targets_test, actual_returns_test)
        ]

        for r in results:
            # A model is 'Stable' if the test accuracy is close to the walk-forward accuracy
            stability = abs(r['accuracy'] - r['walk_forward_accuracy'])
            r["stability"] = stability

        # Save winning model data
        self._save_winning_strategy_assets(ticker, interval, results, features_train, train_data)

        return True

############################################################################

# Save prediction to ledger
def save_prediction(ticker: str, interval: str, current_date: datetime, forecast_results: dict) -> None:
    # Check if that exact prediction has already been saved (note: using same model will always return the same prediction for the same data)
    if prediction_saved(ticker, interval, current_date): return
    if len(forecast_results) < 1: return

    # Iterate through the 3 horizons the model predicted and extract the necessary information
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    new_entries = []
    for step, data in forecast_results.items():
        new_entries.append({
            "Interval": interval,
            'Open_Date': current_date.strftime("%Y-%m-%d %H:%M"),
            'Target_Date': data['target_date'].strftime('%Y-%m-%d %H:%M'),
            'Horizon': f"{data['time_difference']}{interval[1]}",
            "Current_Price": round(data['current_price'], 2),
            'Predicted_Price': round(data['price'], 2),
            'Predicted_Max': round(data['up'], 2),
            'Predicted_Min': round(data['lo'], 2),
            'Direction': data['dir'],
            'Probability': f"{data['probability']:.1%}",
            'Actual_Price': np.nan,
            'Is_Correct': np.nan
        })

    # Add prediction data to the ledger
    df_new = pd.DataFrame(new_entries)
    if not os.path.exists(ledger_file): df_new.to_csv(ledger_file, index=False)
    else: df_new.to_csv(ledger_file, mode='a', header=False, index=False)

# Load prediction from ledger
def load_prediction(ticker: str, interval: str, date: datetime) -> dict:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    ledger = pd.read_csv(ledger_file)
    date = date.strftime("%Y-%m-%d %H:%M")

    # Filter for the specific data
    match = ledger[(ledger['Interval'] == interval) & (ledger['Open_Date'] == date)]
    match_dicts = match.reset_index().to_dict(orient='records')

    # Rebuild the forecast_results dict
    forecast_results = {}
    for i, horizon in zip(range(0,3), [1,5,21]):
        forecast_results[horizon] = {
            "current_price": float(match_dicts[i]['Current_Price']),
            'price': float(match_dicts[i]['Predicted_Price']),
            'up': float(match_dicts[i]['Predicted_Max']),
            'lo': float(match_dicts[i]['Predicted_Min']),
            'target_date': pd.to_datetime(match_dicts[i]['Target_Date'], format='ISO8601'),
            'probability': float(match_dicts[i]['Probability'].replace("%", "")) / 100.0,
            'dir': match_dicts[i]['Direction'],
        }
    return forecast_results

# Checks if there exists an entry in the ledger for that time
def prediction_saved(ticker: str, interval: str, date) -> bool:
    ledger_file = os.path.join(LEDGER_DIR, f"{ticker}_ledger.csv")
    if not os.path.exists(ledger_file): return False

    ledger = pd.read_csv(ledger_file)
    ledger['Open_Date'] = pd.to_datetime(ledger['Open_Date'], format='ISO8601')

    # Check if any entry matches current ticker and last trade date
    match = ledger[(ledger['Interval'] == interval) & (ledger['Open_Date'] == date)]
    return not match.empty

# Run all helper functions to display a prediction
def run_prediction_pipline(ticker: str, interval: str) -> dict:
    # Add in technical indicators
    processed_df, assets = prepare_prediction_data(ticker, interval)
    if any(v is None for v in [processed_df, assets]): return {}

    last_trade_date = processed_df.index[-1]

    # Load or create and save the prediction
    if not prediction_saved(ticker, interval, last_trade_date):
        # Create a dict with basic information about the state of the prediction and stock
        is_hour = "h" in interval
        tech_info = (
            {1:1, 2:2, 4:4, 8:25} if is_hour else {1:1, 2:2, 5:7, 21:28}, # horizons
            "h" if is_hour else "d", # period
            last_trade_date,
            float(processed_df['Close'].iloc[-1]) # current_price
        )

        # Generate a prediction
        forecast_results = generate_forecasts(processed_df, assets, tech_info)
        save_prediction(ticker, interval, last_trade_date, forecast_results)

    else:
        forecast_results = load_prediction(ticker, interval, last_trade_date)

    return forecast_results

# Adds in technical indicators, trains model if needed or loads it
def prepare_prediction_data(ticker: str, interval: str) -> tuple:
    model_path = os.path.join(MODEL_DIR, f"{ticker}_{interval}")

    # Ensure data exists
    df = load_data(ticker, interval)
    if df is None: print("No data"); return None, None

    # If its hourly data, and market has just opened, add in the day opening price
    now_utc = pd.Timestamp.now(tz='UTC')
    schedule = NYSE_CAL.schedule(start_date=now_utc, end_date=now_utc)
    if not schedule.empty:
        market_open = schedule.iloc[0]['market_open']
        if interval == "1h" and market_open <= now_utc <= (market_open + timedelta(hours=1)):

            temp_data = yf.download(ticker, period="1d", interval="1d", progress=False)
            today_open = float(temp_data['Open'].values[-1][0])

            new_row = pd.DataFrame({
                'Open': [today_open], 'High': [today_open],
                'Low': [today_open], 'Close': [today_open], 'Volume': [0], "Adj Close": [today_open]
            }, index=[(market_open-timedelta(hours=1)).tz_localize(None)])

            df = pd.concat([df, new_row])

    # Trains a model if needed
    if not all(os.path.exists(os.path.join(model_path, f)) for f in ["models.pkl", "features.pkl", "scaler.pkl", "metadata.json"]):
        # print(f"Empty or missing trained models found for {ticker}. Training...")
        if not TrainingManager().run_training_pipeline(ticker, interval): return None, None

    # Load assets
    processed_df = TrainingManager().calculate_technical_indicators(df.copy(), ticker, interval, training=False)
    if processed_df.empty: return None, None
    scaler = joblib.load(f"{model_path}/scaler.pkl")
    features = joblib.load(f"{model_path}/features.pkl")

    return processed_df, (scaler, features, model_path)

def get_market_dates(latest_date, horizons, period):
    market_targets = {}

    # latest_date is the OPENING time
    if latest_date.tz is None:
        latest_date = latest_date.tz_localize('UTC')

    end_search = latest_date + pd.Timedelta(days=30)
    schedule = NYSE_CAL.schedule(start_date=latest_date, end_date=end_search)

    if period == "d":
        # For daily: List of DAYs (e.g. '2026-03-06')
        valid_times = schedule.index.normalize()
    else:
        # For hourly: List of CLOSING times of the HOUR (e.g. 15:30 to 21:00)
        valid_times = mcal.date_range(schedule, frequency="1h")

    for step, time_dif in horizons.items():
        if period == "h":
            target_dt = latest_date + timedelta(hours=(time_dif+1))

            if target_dt not in valid_times:
                if time_dif in [4, 25] and (target_dt - timedelta(hours=1)) in valid_times:
                    target_dt = target_dt - timedelta(hours=1)
                else: target_dt = None

        else:
            target_dt = latest_date + timedelta(days=time_dif)
            if target_dt.weekday() == 5:
                target_dt += timedelta(days=2)

            # Holiday
            if target_dt.strftime('%Y-%m-%d') not in valid_times:
                target_dt = None

        market_targets[step] = target_dt

    return market_targets

# Predict the price movement
def generate_forecasts(processed_df: pd.DataFrame, assets: tuple, tech_info: tuple) -> dict:
    scaler, features, model_path = assets
    horizons, period, last_trade_date, current_price =  tech_info

    model_bundle = joblib.load(f"{model_path}/models.pkl")
    scaled_row = scaler.transform(processed_df[features].iloc[-1:])

    current_volatility_atr = float(processed_df['ATR'].iloc[-1])
    forecast_results = {}

    target_dates = get_market_dates(last_trade_date, horizons, period)
    if len(target_dates) < 1: return {}

    # Calculate forecastSs
    for step, actual_time in horizons.items():
        if target_dates[step] is None: continue
        # Load the specific model for this timeframe
        directional_classifier = model_bundle["classifiers"][step]

        # Calculate whether it will go up or down
        up_probability = float(directional_classifier.predict_proba(scaled_row)[0][1])
        adjusted_probability = up_probability if up_probability > 0.5 else 1 - up_probability
        direction = "UP ▲" if up_probability > 0.5 else "DOWN ▼"

        # Calculate predicted price
        direction_multiplier = 1 if up_probability > 0.5 else -1
        confidence_strength = 2 * (adjusted_probability - 0.5)
        expected_move_magnitude = current_volatility_atr * np.sqrt(step)

        predicted_price = current_price + (direction_multiplier * expected_move_magnitude * confidence_strength)

        capped_width = min(expected_move_magnitude * (1.0 + confidence_strength), current_price * 0.15)

#         print(f"""
# NVDA: {actual_time}{period}
# Latest Date: {last_trade_date}
# Target Date: {target_dates[step]}
# Current price: {current_price}
# Up probability: {up_probability}
# Predicted price: {predicted_price}
# Predicted low: {predicted_price - capped_width}
# Predicted high: {predicted_price + capped_width}
# """)

        forecast_results[step] = {
            "current_price": current_price,
            'price': predicted_price,
            'up': predicted_price + capped_width,
            'lo': predicted_price - capped_width,
            'target_date': target_dates[step],
            'time_difference': actual_time,
            'probability': adjusted_probability,
            'dir': direction
        }

    return forecast_results

############################################################################

