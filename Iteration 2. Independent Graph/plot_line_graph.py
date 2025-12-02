import sys
import yfinance as yf
import pyqtgraph as pg
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from datetime import datetime
import os
import pandas as pd

# --- 1. Data Configuration ---
TICKER = "AAPL"
START_DATE = "2000-01-01"
END_DATE = "2025-01-12"
PLOT_COLUMN = 'Close' 
HOVER_THRESHOLD = 10  # Pixels - how close to the line to trigger crosshair
CACHE_DIR = "stock_data_cache"
SAVE_DATA = True  # Set to False to skip saving


class Graph:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.start_date = "2000-01-01" # Will be as late as possible
        self.end_date = "2025-01-12"
        self.data_dir = "stock_data_cache"

        close_data = self.load_data(self.ticker)

        plot_stock_data(close_data)


    def load_data(self, ticker):
        cache_file = os.path.join("stock_data_cache", f"{ticker}.csv")
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            return data["Close"]

        print(f"Downloading {ticker} data...") # temp Debug


        data = yf.download(ticker, start="2000-01-01", end="2025-01-12", progress=False) # dates temp

        # debugs
        if data.empty:
            print(f"yfinance returned an empty dataset for {ticker}.")
            return None
        if "Close" not in data.columns:
            print(f"Error: 'Close' column not found. Available columns: {list(data.columns)}")
            return None

        if not os.path.exists("stock_data_cache"):
            os.makedirs("stock_data_cache")
        try:
            data.to_csv(cache_file)
            print(f"Data saved to {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

        return data["Close"]

def get_cache_filename(ticker, start, end):
    # Generate a cache filename based on ticker and date range
    return os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")

def load_cached_data(ticker, start, end):
    # Try to load data from cache
    cache_file = get_cache_filename(ticker, start, end)
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached data from {cache_file}...")
            data = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            return data[PLOT_COLUMN]
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
            
        if PLOT_COLUMN not in data.columns:
            print(f"Error: '{PLOT_COLUMN}' column not found. Available columns: {list(data.columns)}")
            return None
        
        result = data[PLOT_COLUMN]
        
        if SAVE_DATA:
            save_data_to_cache(data, ticker, start, end)
        
        return result
        
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None

def plot_stock_data(data):
    """
    Plots the stock data using PyQtGraph with a crosshair that appears only when hovering over the line.
    """
    
    # 1. Prepare Data
    dates_in_seconds = data.index.to_numpy().astype(np.int64) // 10**9
    prices = data.values.astype(float).flatten()

    # --- 2. PyQtGraph Setup ---
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle(f'{TICKER} Stock Price - PyQtGraph (Interactive)')

    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)

    date_axis = pg.DateAxisItem(orientation='bottom')
    plot_widget = pg.PlotWidget(axisItems={'bottom': date_axis})
    layout.addWidget(plot_widget)

    # Plot the data
    curve = plot_widget.plot(
        dates_in_seconds, 
        prices,           
        pen=pg.mkPen(color='#3498db', width=2), 
        name=PLOT_COLUMN
    )
    
    plot_widget.setTitle(f'{PLOT_COLUMN} Price for {TICKER}')
    plot_widget.setLabel('left', 'Price', units='USD')
    plot_widget.setLabel('bottom', 'Date')
    plot_widget.showGrid(x=True, y=True)
    
    # --- 3. X-Axis Stretching Setup ---
    
    # Store state for x-axis stretching
    stretch_state = {'dragging': False, 'start_x': None, 'start_range': None}
    
    plot_item = plot_widget.getPlotItem()
    view_box = plot_item.vb
    original_mouseDragEvent = view_box.mouseDragEvent
    
    def custom_mouseDragEvent(ev, axis=None):
        """Custom drag handler to detect bottom-of-graph drags for x-axis stretching."""
        try:
            if ev.button() == 1:  # Left mouse button
                # Get the mouse position in view coordinates
                pos = ev.pos()
                
                # Get the plot area bounds in scene coordinates
                view_rect = view_box.sceneBoundingRect()
                
                # Check if we're near the bottom (within 30 pixels)
                bottom_threshold = 100
                if pos.y() > view_rect.bottom() - bottom_threshold:
                    # X-axis stretching mode
                    if ev.isStart():
                        stretch_state['dragging'] = True
                        stretch_state['start_x'] = pos.x()
                        stretch_state['start_range'] = view_box.viewRange()
                    
                    if stretch_state['dragging'] and not ev.isFinish():
                        # Calculate the drag distance in screen coordinates
                        delta = pos.x() - stretch_state['start_x']
                        
                        # Get current view range
                        x_range = stretch_state['start_range'][0]
                        y_range = stretch_state['start_range'][1]
                        
                        view_width = view_rect.width()
                        data_width = x_range[1] - x_range[0]
                        
                        # Adjust x range based on drag direction
                        stretch_factor = 1 - (delta / view_width) * 0.5
                        stretch_factor = max(0.1, min(2.0, stretch_factor))
                        
                        new_width = data_width / stretch_factor
                        center_x = (x_range[0] + x_range[1]) / 2
                        
                        new_x_range = [center_x - new_width / 2, center_x + new_width / 2]
                        view_box.setRange(xRange=new_x_range, yRange=y_range, padding=0)
                    
                    if ev.isFinish():
                        stretch_state['dragging'] = False
                    
                    ev.accept()
                    return
            
            # Normal PyQtGraph behavior for other areas
            original_mouseDragEvent(ev, axis)
        except Exception as e:
            print(f"Error in custom_mouseDragEvent: {e}")
            original_mouseDragEvent(ev, axis)
    
    view_box.mouseDragEvent = custom_mouseDragEvent
    
    # --- 4. Crosshair Implementation ---
    
    vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=1.5))
    hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r', width=1.5))
    plot_widget.addItem(vLine, ignoreBounds=True)
    plot_widget.addItem(hLine, ignoreBounds=True)
    
    # Hide lines initially
    vLine.hide()
    hLine.hide()

    coord_label = pg.TextItem(
        text="Date: N/A, Price: N/A",
        color=(200, 200, 200),
        anchor=(1, 1)
    )
    plot_widget.addItem(coord_label, ignoreBounds=True)
    coord_label.hide()
    
    def is_near_curve(mouse_point_view, threshold_pixels=HOVER_THRESHOLD):
        """
        Check if mouse_point_view is within threshold_pixels of the curve.
        Returns (is_close, closest_price) where closest_price is the y-value on the curve at that x.
        """
        mouse_x = mouse_point_view.x()
        mouse_y = mouse_point_view.y()
        
        # Find the closest data point to the mouse x-coordinate
        distances = np.abs(dates_in_seconds - mouse_x)
        closest_idx = np.argmin(distances)
        
        # Get the curve's y-value at that x position
        curve_y = prices[closest_idx]
        
        # Convert pixel threshold to data coordinates
        # Get the bounds of the plot in view coordinates
        plot_item = plot_widget.getPlotItem()
        view_box = plot_item.vb
        
        # Get a small offset in pixels and convert to data coordinates
        ref_point = view_box.mapSceneToView(plot_widget.mapToScene(0, 0))
        offset_point = view_box.mapSceneToView(plot_widget.mapToScene(threshold_pixels, 0))
        pixel_to_data = abs(offset_point.x() - ref_point.x())
        
        # Check if mouse is close in both x and y
        x_dist = abs(dates_in_seconds[closest_idx] - mouse_x)
        y_dist = abs(curve_y - mouse_y)
        
        # Convert y_dist to pixel equivalent
        ref_y = view_box.mapSceneToView(plot_widget.mapToScene(0, 0)).y()
        offset_y = view_box.mapSceneToView(plot_widget.mapToScene(0, threshold_pixels)).y()
        pixel_to_data_y = abs(offset_y - ref_y)
        
        is_close = (x_dist < pixel_to_data * 2) and (y_dist < pixel_to_data_y * 2)
        
        return is_close, curve_y, closest_idx
    
    def mouseMoved(pos):
        """Handler for mouse movement over the plot."""
        if plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = plot_widget.getPlotItem().vb.mapSceneToView(pos)
            
            is_close, curve_y, closest_idx = is_near_curve(mousePoint)
            
            if is_close:
                # Show crosshair and label
                vLine.show()
                hLine.show()
                coord_label.show()
                
                # Update line positions
                vLine.setPos(dates_in_seconds[closest_idx])
                hLine.setPos(curve_y)
                
                # Format and update label
                timestamp_sec = dates_in_seconds[closest_idx]
                try:
                    date_str = datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d')
                except ValueError:
                    date_str = "Invalid Date"
                
                price_str = f"{curve_y:.2f}"
                coord_label.setText(f"Date: {date_str}, Price: ${price_str}")
                coord_label.setPos(dates_in_seconds[closest_idx], curve_y)
            else:
                # Hide crosshair when not near the curve
                vLine.hide()
                hLine.hide()
                coord_label.hide()

    plot_widget.scene().sigMouseMoved.connect(mouseMoved)

    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    ticker = input("Ticker: ") # Will be taken from box input
    MainGraph = Graph(ticker)


    # aapl_data = get_stock_data(TICKER, START_DATE, END_DATE)
    #
    # if aapl_data is not None and not aapl_data.empty:
    #     plot_stock_data(aapl_data)
    # else:
    #     print("Exiting plot due to data error.")