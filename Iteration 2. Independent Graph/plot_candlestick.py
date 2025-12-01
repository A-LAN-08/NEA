import sys
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtChart import QChart, QChartView, QCandlestickSeries, QCandlestickSet, QDateTimeAxis, QValueAxis
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtGui import QColor, QPainter

# --- 1. Data Configuration ---
TICKER = "AAPL"
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
CACHE_DIR = "stock_data_cache"
SAVE_DATA = True

def get_cache_filename(ticker, start, end):
    """Generate a cache filename based on ticker and date range."""
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")

def load_cached_data(ticker, start, end):
    """Try to load data from cache."""
    cache_file = get_cache_filename(ticker, start, end)
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached data from {cache_file}...")
            data = pd.read_csv(cache_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            return data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None

def save_data_to_cache(data, ticker, start, end):
    """Save data to cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_file = get_cache_filename(ticker, start, end)
    try:
        data.to_csv(cache_file)
        print(f"Data saved to {cache_file}")
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_stock_data(ticker, start, end):
    """Downloads stock data from yfinance or loads from cache."""
    # Try loading from cache first
    cached_data = load_cached_data(ticker, start, end)
    if cached_data is not None:
        return cached_data
    
    print(f"Downloading {ticker} data from {start} to {end}...")
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if data.empty:
            print(f"yfinance returned an empty dataset for {ticker}.")
            return None
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            print(f"Error: Missing columns {missing}. Available columns: {list(data.columns)}")
            return None
        
        if SAVE_DATA:
            save_data_to_cache(data, ticker, start, end)
        
        return data
        
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

def plot_candlestick(data):
    """Plot candlestick chart using QtCharts."""
    
    app = QApplication(sys.argv)
    
    # Create series
    series = QCandlestickSeries()
    series.setName(TICKER)
    series.setIncreasingColor(QColor(Qt.green))
    series.setDecreasingColor(QColor(Qt.red))
    
    # Add candlestick data
    for date, row in data.iterrows():
        timestamp = int(date.timestamp() * 1000)  # Convert to milliseconds
        
        candlestick_set = QCandlestickSet(timestamp)
        candlestick_set.setOpen(row['Open'].item())
        candlestick_set.setHigh(row['High'].item())
        candlestick_set.setLow(row['Low'].item())
        candlestick_set.setClose(row['Close'].item())
        
        series.append(candlestick_set)
    
    # Get date range for axes
    first_date = data.index.min()
    last_date = data.index.max()
    
    # Create chart
    chart = QChart()
    chart.addSeries(series)
    chart.setTitle(f"{TICKER} Candlestick Chart")
    chart.setAnimationOptions(QChart.SeriesAnimations)
    
    # Create custom axes
    axis_x = QDateTimeAxis()
    axis_x.setFormat("MMM yyyy")
    axis_x.setTitleText("Date")
    axis_x.setRange(QDateTime.fromMSecsSinceEpoch(int(first_date.timestamp() * 1000)),
                    QDateTime.fromMSecsSinceEpoch(int(last_date.timestamp() * 1000)))
    
    axis_y = QValueAxis()
    axis_y.setTitleText("Price (USD)")
    min_price = data[['Low']].min().values[0]
    max_price = data[['High']].max().values[0]
    price_range = max_price - min_price
    
    # Calculate nice round tick interval
    def get_nice_interval(range_val):
        """Calculate a nice round interval for axis ticks."""
        if range_val == 0:
            return 1
        magnitude = 10 ** (int(len(str(int(range_val)))) - 1)
        normalized = range_val / magnitude
        
        if normalized <= 1.5:
            return magnitude / 10
        elif normalized <= 3:
            return magnitude / 5
        elif normalized <= 7:
            return magnitude / 2
        else:
            return magnitude
    
    tick_interval = get_nice_interval(price_range)
    tick_count = max(4, int(price_range / tick_interval) + 1)
    
    axis_y.setRange(int(min_price - tick_interval), int(max_price + tick_interval))
    axis_y.setTickCount(tick_count)
    axis_y.setLabelFormat("%.2f")
    
    chart.addAxis(axis_x, Qt.AlignBottom)
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_x)
    series.attachAxis(axis_y)
    
    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignBottom)
    
    # Create chart view with interactive features
    chart_view = QChartView(chart)
    chart_view.setRenderHint(QPainter.Antialiasing)
    
    # Create window
    window = QMainWindow()
    window.setCentralWidget(chart_view)
    window.setWindowTitle(f"{TICKER} Candlestick Chart")
    window.resize(1200, 600)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    stock_data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    if stock_data is not None and not stock_data.empty:
        plot_candlestick(stock_data)
    else:
        print("Exiting plot due to data error.")