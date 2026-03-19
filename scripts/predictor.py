
# Standard library imports
import json
import os
from pathlib import Path
import shutil
import warnings
from datetime import datetime, timedelta

# External library imports
import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import talib
from pykalman import KalmanFilter
import yfinance as yf
from lightgbm import LGBMClassifier
from lightgbm import Booster as LGBMBooster
from PyQt6.QtCore import QThread, pyqtSignal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier, dataset
from skorch.callbacks import EarlyStopping, EpochScoring
from safetensors.torch import save_file, load_file


# Set environment variables and filters
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")
NYSE_CAL = mcal.get_calendar('NYSE')

# Custom imports
from scripts.data_management import load_data
from scripts.config import LEDGER_DIR, MODEL_DIR, DATA_DIR

class Settings:
    VERBOSE = 0 # Set whether to display logging or not

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
            forecast_results = run_prediction_pipeline(self.ticker, self.interval)
            self.training_finished.emit(forecast_results)
        except Exception as e:
            self.training_error.emit(str(e))

class LSTMBrain(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x) # We only want the 'final thought' of the last layer
        return self.sigmoid(self.fc(hn[-1])).squeeze(-1)

class LSTM:
    def __init__(self, seed: int):
        self.seed = seed

    def train(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                    target_test: pd.Series, price_returns: np.ndarray) -> dict:

        data_normalizer = StandardScaler()

        # Scale the data first
        features_train_normalized = data_normalizer.fit_transform(features_train)
        features_test_normalized = data_normalizer.transform(features_test)

        # Walk-Forward Validation
        time_splitter = TimeSeriesSplit(n_splits=3)
        validation_scores = []

        for train_idx, val_idx in time_splitter.split(features_train_normalized):
            # Create sequences for this specific fold
            x_fold_train, y_fold_train = self.create_3d_sequences(features_train_normalized[train_idx], target_train.iloc[train_idx].values)
            x_fold_val, y_fold_val = self.create_3d_sequences(features_train_normalized[val_idx], target_train.iloc[val_idx].values)

            if len(x_fold_train) == 0 or len(x_fold_val) == 0: continue

            fold_model = self.get_lstm_competitor(input_dim=x_fold_train.shape[2])
            fold_model.fit(x_fold_train, y_fold_train)
            validation_scores.append(fold_model.score(x_fold_val, y_fold_val))

        # Turn into 3D "Movies" (14-day lookback)
        x_train_3d, y_train_3d = self.create_3d_sequences(features_train_normalized, target_train.values)
        x_test_3d, y_test_3d = self.create_3d_sequences(features_test_normalized, target_test.values)

        # Train
        model = self.get_lstm_competitor(input_dim=x_train_3d.shape[2])
        model.fit(x_train_3d, y_train_3d)

        test_predictions = model.predict(x_test_3d)

        accuracy, sharpe, abs_sharpe, needs_flip = TrainingManager.evaluate_performance(
            target_test.iloc[14:],
            test_predictions,
            price_returns[14:]
        )

        return {
            'model_type': 'LSTM',
            'accuracy': accuracy,
            'walk_forward_accuracy': np.mean(validation_scores) if validation_scores else 0,
            'sharpe_ratio': sharpe,
            'absolute_sharpe': abs_sharpe,
            'logic_flipped': needs_flip,
            'raw_predictions': test_predictions,
            'trained_model_object': model,
            'feature_scaler': data_normalizer
        }

    @staticmethod
    def get_lstm_competitor(input_dim):
        return NeuralNetClassifier(
            LSTMBrain,
            module__input_dim=input_dim,
            max_epochs=100,
            lr=0.001, # learning rate
            train_split=dataset.ValidSplit(0.2, stratified=False),
            iterator_train__shuffle=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=Settings.VERBOSE,
            criterion=nn.BCELoss,
            callbacks=[
                ('early_stopping', EarlyStopping(
                    monitor='valid_loss',
                    patience=5,  # Stop if no improvement after 5 epochs
                    lower_is_better=True
                )),
                ('val_acc', EpochScoring(
                    scoring='accuracy',
                    name='valid_acc',
                    lower_is_better=False
                ))
            ]
        )

    @staticmethod
    def create_3d_sequences(data, targets, window_size=14):
        x, y = [], []
        for i in range(len(data) - window_size):
            x.append(data[i: i + window_size])
            y.append(targets[i + window_size])
        return np.array(x).astype(np.float32), np.array(y).astype(np.float32)

# Class to control and train models
class TrainingManager:
    def __init__(self):
        self.seed = 69
        self.sharpe_threshold = 0.50 # Min sharpe value for model to be useful
        self.__test_size = 0.2 # How much of data used to test vs train

    @staticmethod
    def _integrate_sentiment(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        target_file = os.path.join(DATA_DIR, f"master_sentiment.parquet")

        sent_df = pd.read_parquet(target_file, filters=[('ticker', '==', ticker)])

        sent_df['event_date'] = pd.to_datetime(sent_df['event_date'])
        sent_df = sent_df.sort_values('event_date')

        # Sentiment Impact: tone * log(count + 1)
        sent_df['sentiment_impact'] = sent_df['avg_tone'] * np.log1p(sent_df['article_count'])

        # Sentiment simple moving averages
        sent_df['sentiment_sma_7d'] = sent_df['avg_tone'].rolling(7, min_periods=1).mean()
        sent_df['sentiment_sma_30d'] = sent_df['avg_tone'].rolling(30, min_periods=1).mean()

        # Sentiment Volatility (Rolling Standard Deviation)
        sent_df['sentiment_volatility_7d'] = sent_df['avg_tone'].rolling(7, min_periods=1).std().fillna(0)

        # Sentiment Momentum (The gap between short and long term vibes)
        # Positive = Sentiment is improving; Negative = Sentiment is cooling off
        sent_df['sentiment_momentum'] = sent_df['sentiment_sma_7d'] - sent_df['sentiment_sma_30d']

        # Sentiment volume Z-Score (The "Shock" factor)
        # This identifies days when news volume is significantly higher than usual for THAT specific ticker
        sent_df['sentiment_volume_zscore'] = (
                sent_df['article_count'] / (sent_df['article_count'].rolling(30).mean() + 1e-9)
        ).fillna(0)

        # Other sentiment features
        sent_df['sentiment_shock'] = sent_df['avg_tone'].diff().fillna(0)
        sent_df['negative_pressure'] = (
            np.abs(np.minimum(sent_df['avg_tone'], 0)) * np.log1p(sent_df['article_count'])
        )
        sent_df['days_since_news'] = (sent_df['article_count'] > 0).astype(int).groupby((sent_df['article_count'] > 0).cumsum()).cumcount()

        sent_df['merge_date'] = sent_df['event_date'] + pd.Timedelta(days=1)
        sent_df = sent_df[sent_df.columns.difference(['ticker', 'event_date'])]

        index_name = df.index.name if df.index.name else 'index'
        df['merge_date'] = pd.to_datetime(df.index).normalize()
        df = df.reset_index()

        df = pd.merge(df, sent_df, on='merge_date', how='left')
        df = df.set_index(index_name)
        df = df.drop(columns=['merge_date']).fillna(0)

        return df

    @staticmethod
    def calculate_hurst(series, window=100):
        if len(series) < window: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) + 1e-9 for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    # Calculate technical indicators
    def calculate_technical_indicators(self, df: pd.DataFrame, ticker: str, interval: str, training: bool = True) -> pd.DataFrame:
        df = self._integrate_sentiment(df, ticker)

        # Horizon (h) targets for selected period (p)
        p = "h" if "h" in interval else "d"
        for h in ([1,2,4,8] if "h" in interval else [1,2,5,21]):
            df[f'target_cls_{h}{p}'] = df['Adj Close'].shift(-h).gt(df['Adj Close']).astype(int)

        df['return'] = df['Adj Close'].pct_change()
        for i in range(1, 4): df[f'return_lag_{i}'] = df['return'].shift(i)

        # Technical indicators
        df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
        macd, _, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD_Hist'] = macdhist
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
        df['MA_200'] = talib.SMA(df['Adj Close'], timeperiod=200)
        df['PDMA_200'] = (df['Adj Close'] / df['MA_200']) - 1

        df['OBV'] = talib.OBV(df['Adj Close'], df['Volume'])
        upper, mid, lower = talib.BBANDS(df['Adj Close'], timeperiod=20)
        df['BBP'] = (df['Adj Close'] - lower) / (upper - lower)
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)

        # Deviations using all of OHLC
        df['range_pct'] = (df['High'] - df['Low']) / df['Close']
        df['body_pct'] = (df['Close'] - df['Open']) / df['Close']
        df['upper_shadow_pct'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['lower_shadow_pct'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

        # Hurst Exponent
        df['Hurst_Exponent'] = df['Close'].rolling(window=100, min_periods=100).apply(self.calculate_hurst, raw=True)
        df['Hurst_Exponent'] = df['Hurst_Exponent'].fillna(0.5)

        # Kalman Filter
        def get_kalman_filter(series):
            kf = KalmanFilter(transition_matrices=[1],
                              observation_matrices=[1],
                              initial_state_mean=series.iloc[0],
                              initial_state_covariance=1,
                              observation_covariance=1,
                              transition_covariance=0.01)
            state_means, _ = kf.filter(series.values)
            return state_means.flatten()

        df['Kalman_Price'] = get_kalman_filter(df['Close'])
        df['Kalman_Dev'] = (df['Close'] - df['Kalman_Price']) / df['Kalman_Price']  # Deviation from "True" price

        # Efficiency ratio
        price_diff = df['Close'].diff(20).abs()
        volatility = df['Close'].diff().abs().rolling(20).sum()
        df['Efficiency_Ratio'] = price_diff / volatility  # 1.0 = Strong Trend, 0.0 = Choppy/Noisy

        # Market context
        spy_data = pd.read_parquet(os.path.join(DATA_DIR, f'SPY_{interval}.parquet'))
        spy_data.index.name = "Date"
        spy_data.index = pd.to_datetime(spy_data.index, utc=True).tz_localize(None)
        spy_data = spy_data[~spy_data.index.duplicated(keep='first')]

        aligned_market = spy_data.reindex(df.index).ffill()
        market_returns = aligned_market['Close'].pct_change()
        stock_returns = df['return']

        # Rolling Beta (60-period): Covariance(stock, market) / Variance(market)
        rolling_cov = stock_returns.rolling(window=60).cov(market_returns)
        rolling_var = market_returns.rolling(window=60).var()
        df['Market_Beta'] = (rolling_cov / rolling_var + 1e-9).fillna(1.0)  # Assume 1.0 if no data

        # Relative Strength: Ratio of stock price to market price (normalized)
        df['Relative_Strength'] = (df['Adj Close'] / aligned_market['Close']).pct_change().fillna(0)

        # Other indicators
        df['vol_ratio'] = df["return"].rolling(5).std() / df["return"].rolling(50).std()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        if training:
            df = df.dropna()

        return df

    # Evaluate model performance with accuracy and sharpe ratio
    @staticmethod
    def evaluate_performance(actual_direction: pd.Series, predicted_direction: np.ndarray, actual_returns: np.ndarray):
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
        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=self.seed,
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
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions,price_returns)
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
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions, price_returns)
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
        accuracy, sharpe, abs_sharpe, needs_flip = self.evaluate_performance(target_test, test_predictions, price_returns)
        # Return all the information from the model
        return {
            'model_type': 'SVC', 'accuracy': accuracy, 'walk_forward_accuracy': np.mean(validation_scores),
            'sharpe_ratio': sharpe, 'absolute_sharpe': abs_sharpe, 'logic_flipped': needs_flip,
            'raw_predictions': test_predictions, 'trained_model_object': model, 'feature_scaler': data_normalizer}

    def _train_lstm(self, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame,
                    target_test: pd.Series, price_returns: np.ndarray) -> dict:

        lstm_model = LSTM(self.seed)
        return lstm_model.train(features_train, target_train, features_test, target_test, price_returns)

    # Save models for all horizons with the best performing model type
    def _save_winning_strategy_assets(self, ticker: str, interval: str, full_results: dict,
                                      features_data: pd.DataFrame, targets_dataframe: pd.DataFrame) -> None:
        # Create the folder for this specific stock's model data to save
        save_folder = os.path.join(MODEL_DIR, f"{ticker}_{interval}")
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        scaler = StandardScaler().fit(features_data)
        scaled_features = scaler.transform(features_data)

        meta_full = {}
        for h, horizon_results in full_results.items():
            best_model_type = max(horizon_results, key=lambda x: x["absolute_sharpe"])["model_type"]
            meta_full[str(h)] = {
                "best model": best_model_type,
                "training date": datetime.now().strftime("%Y-%m-%d"),
                "full results": [
                    {k: (v.item() if hasattr(v, 'item') else v)
                     for k, v in result.items()
                     if k not in {"raw_predictions", "trained_model_object", "feature_scaler"}}
                    for result in full_results[h]
                ],
            }

            target_col = f'target_cls_{h}{interval[1]}'
            valid_idx = targets_dataframe[target_col].notna()
            x_final = scaled_features[valid_idx.values]
            y_final = targets_dataframe[target_col][valid_idx]

            model_file = f"model_{h}{interval[1]}"

            # Train Classifier
            if best_model_type == 'LSTM':
                lstm_base = LSTM(self.seed)
                x_3d, y_3d = lstm_base.create_3d_sequences(x_final, y_final)

                model = lstm_base.get_lstm_competitor(input_dim=x_3d.shape[2])
                model.fit(x_3d, y_3d)

                save_file(model.module_.state_dict(), os.path.join(save_folder, f"{model_file}.safetensors"))
                meta_full[str(h)]["file_format"] = ".safetensors"

            elif best_model_type == 'LGBM':
                up_days = y_final.sum()
                down_days = len(y_final) - up_days
                spw = down_days / up_days if up_days > 0 else 1.0

                model = LGBMClassifier(
                    n_estimators=100, learning_rate=0.05, random_state=self.seed,
                    scale_pos_weight=spw, verbose=-1, device="gpu", gpu_platform_id=0, gpu_device_id=0
                )
                model.fit(x_final, y_final)

                model.booster_.save_model(os.path.join(save_folder, f"{model_file}.txt"))
                meta_full[str(h)]["file_format"] = ".txt"

            else:
                if best_model_type == 'Lasso':
                    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=self.seed, class_weight='balanced')
                else:  # SVC
                    model = SVC(kernel='rbf', C=1.0, random_state=self.seed, class_weight='balanced', probability=True)

                model.fit(x_final, y_final)

                joblib.dump(model, os.path.join(save_folder, f"{model_file}.joblib"))
                meta_full[str(h)]["file_format"] = ".joblib"

        with open(os.path.join(save_folder, 'metadata.json'), 'w') as f:
            json.dump(meta_full, f, indent=4)
        joblib.dump(scaler, os.path.join(save_folder, "scaler.joblib"))
        joblib.dump(list(features_data.columns), os.path.join(save_folder, "features.joblib"))

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
        drop_columns = [c for c in df.columns if 'target' in c or c in ['Open', 'High', 'Low', 'Close', "Adj Close", 'Volume', 'MA_200', 'return']]
        train_columns = [c for c in df.columns if c not in drop_columns]

        # Create input features and answers
        period = "h" if 'h' in interval else "d"
        features_train, features_test = train_data[train_columns], test_data[train_columns]

        results = {}

        for h in ([1,2,4,8] if period == "h" else [1,2,5,21]):

            targets_train, targets_test = train_data[f'target_cls_{h}{period}'], test_data[f'target_cls_{h}{period}']
            actual_returns_test = test_data['return'].values

            # print(f"Training for horizon {h}...")
            # Model competition
            horizon_results = [
                self._train_lightgbm(features_train, targets_train, features_test, targets_test, actual_returns_test),
                self._train_lasso_regression(features_train, targets_train, features_test, targets_test, actual_returns_test),
                self._train_support_vector(features_train, targets_train, features_test, targets_test, actual_returns_test),
                self._train_lstm(features_train, targets_train, features_test, targets_test, actual_returns_test)
            ]

            for r in horizon_results:
                # A model is 'Stable' if the test accuracy is close to the walk-forward accuracy
                stability = abs(r['accuracy'] - r['walk_forward_accuracy'])
                r["stability"] = stability

            results[h] = horizon_results

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
    for i, step in enumerate([1, 2, 4, 8] if "h" in interval else [1, 2, 5, 21]):
        forecast_results[step] = {
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
def run_prediction_pipeline(ticker: str, interval: str) -> dict:
    # Add in technical indicators
    processed_df, assets = prepare_prediction_data(ticker, interval)
    if any(v is None for v in [processed_df, assets]):
        print("No data or assets.")
        return {}

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
    existing_stems = {p.stem for p in Path(model_path).iterdir()} if os.path.exists(model_path) else set()
    horizons = [1, 2, 4, 8] if 'h' in interval else [1, 2, 5, 21]
    required_files = {f"model_{h}{interval[1]}" for h in horizons} | {"features", "scaler", "metadata"}
    if not required_files.issubset(existing_stems):
        # print("Empty or missing model files. Training...")
        if not TrainingManager().run_training_pipeline(ticker, interval):
            # print("Failed to train models.")
            return None, None

    # Load assets
    processed_df = TrainingManager().calculate_technical_indicators(df.copy(), ticker, interval, training=False)
    if processed_df.empty:
        # print("Failed to calculate technical indicators.")
        return None, None

    scaler = joblib.load(f"{model_path}/scaler.joblib")
    features = joblib.load(f"{model_path}/features.joblib")

    return processed_df, (scaler, features, model_path)

def get_market_dates(latest_date, horizons, period):
    market_targets = {}

    # latest_date is the OPENING time
    if latest_date.tz is None:
        latest_date = latest_date.tz_localize('UTC')

    end_search = latest_date + pd.Timedelta(days=35)
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

            # Keep rolling forward if it lands on a weekend or holiday
            while target_dt.strftime("%Y-%m-%d") not in valid_times:
                target_dt += timedelta(days=1)

                if (target_dt - latest_date).days > 35:
                    target_dt = None
                    break

        market_targets[step] = target_dt

    return market_targets

# Predict the price movement
def generate_forecasts(processed_df: pd.DataFrame, assets: tuple, tech_info: tuple) -> dict:
    scaler, features, model_folder = assets
    horizons, period, last_trade_date, current_price =  tech_info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(model_folder, 'metadata.json'), 'r') as f:
        meta = json.load(f)

    current_volatility_atr = float(processed_df['ATR'].iloc[-1])
    forecast_results = {}

    target_dates = get_market_dates(last_trade_date, horizons, period)
    if len(target_dates) < 1: return {}

    # Calculate forecastSs
    for step, actual_time in horizons.items():
        if target_dates[step] is None: continue

        # Load the specific model for this timeframe
        ext = meta[str(step)]['file_format']
        model_path = os.path.join(model_folder, f"model_{step}{period}{ext}")

        if ext == ".safetensors":
            recent_data = processed_df[features].tail(14)
            scaled_seq = scaler.transform(recent_data)

            # Convert to 3D: (1, 14, num_features)
            x_3d = np.expand_dims(scaled_seq, axis=0).astype(np.float32)

            brain = LSTMBrain(len(features))
            state_dict = load_file(model_path)
            brain.load_state_dict(state_dict)
            brain.to(device)
            brain.eval()

            with torch.no_grad():
                input_tensor = torch.from_numpy(x_3d)
                up_probability = float(brain(input_tensor).item())

        elif ext == ".txt":
            scaled_row = scaler.transform(processed_df[features].iloc[-1:])
            booster = LGBMBooster(model_file=model_path)
            up_probability = float(booster.predict(scaled_row)[0])

        else: # Joblib (Lasso or SVC)
            scaled_row = scaler.transform(processed_df[features].iloc[-1:])
            model = joblib.load(model_path)
            up_probability = float(model.predict_proba(scaled_row)[0][1])

        # Calculate whether it will go up or down
        adjusted_probability = up_probability if up_probability > 0.5 else 1 - up_probability
        direction = "UP ▲" if up_probability > 0.5 else "DOWN ▼"

        # Calculate predicted price
        direction_multiplier = 1 if up_probability > 0.5 else -1
        confidence_strength = 2 * (adjusted_probability - 0.5)
        expected_move_magnitude = current_volatility_atr * np.sqrt(step)

        predicted_price = current_price + (direction_multiplier * expected_move_magnitude * confidence_strength)
        capped_width = min(expected_move_magnitude * (1.0 + confidence_strength), current_price * 0.15)

#         print(f"""
# Time: {actual_time}{period}
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

