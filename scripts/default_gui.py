
# External library imports
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QSlider, QVBoxLayout, QWidget
)

# Custom imports
from data_management import UpdateManager, abs_file, validate_ticker
from profile_gui import ProfileWindow
from profile_management import DataManager, Profile
from predictor import TrainingWorker
from custom_widgets import CustomButton, add_to_layout, create_circle_label, create_slider_layout
from graph import StockGraph

############################################################################

class MainWindow(QMainWindow):
    # Classes
    logged_profile: Profile | None
    graph: StockGraph
    updater: UpdateManager
    thread: TrainingWorker
    # Containers
    graph_container: QVBoxLayout
    pd_set_frame: QFrame
    # Inputs
    ticker_input: QLineEdit
    ticker_pd_input: QLineEdit
    ticker_list_widget: QComboBox
    type_dropdown: QComboBox
    res_dropdown: QComboBox
    risk_slider: QSlider
    # Displays
    pd_result_label: QLabel
    keys_label: QLabel
    update_label: QLabel
    update_progress: QProgressBar

    def __init__(self):
        # Initialize the main window and dictionaries for button groups
        super().__init__()
        self.setWindowTitle("Stock Prediction App")
        self.setWindowIcon(QIcon(abs_file("stocks.png")))
        self.setGeometry(100, 100, 1500, 900)
        self.setStyleSheet("QWidget {background-color: white; color: black;}")
        self.btns = {"top_btns": [], "pd_type_btns": [], "confirmation_btns": []}

        # Initialize the profile logic
        self.data_manager = DataManager()
        self.logged_in = False
        self.logged_profile = None
        self.status_label = QLabel("Not logged in")

        # Set up the main layout and save to dict for reframing later
        central = QWidget(); self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(); central.setLayout(self.main_layout)

        self.main_frames = {"center": [self.build_center_frame(), 15], "right": [self.build_right_frame(), 3]}
        items, sizes = zip(*self.main_frames.values())
        add_to_layout(self.main_layout, items, size_ratios=sizes)

    # Initialize the center frame with top bar and graph area
    def build_center_frame(self) -> QFrame:
        # Main frame styling
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(0 ,0 ,0 ,0)

        # Top frame styling
        top_layout = QVBoxLayout()
        btn_layout = QHBoxLayout(); btn_layout.setSpacing(4)

        # Ticker input and list
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Enter ticker (e.g. AAPL, TSLA, NVDA)")
        self.ticker_input.returnPressed.connect(self.add_to_graph)
        self.ticker_list_widget = QComboBox()

        # Drop-down lists
        self.type_dropdown = QComboBox(); self.type_dropdown.addItems(["Line", "Candle"])
        self.type_dropdown.currentTextChanged.connect(self.switch_graph_type)

        self.res_dropdown = QComboBox(); self.res_dropdown.addItems(["15m", "1h", "4h", "1d"])
        self.res_dropdown.setCurrentText("1d")
        self.res_dropdown.currentTextChanged.connect(self.switch_graph_res)

        # Add items to top layout
        add_to_layout(btn_layout,
            items=[
                QLabel("Ticker:"), self.ticker_input,
                CustomButton("add_stock_btn", "top_btns", "indv", self, text="Add Ticker", height=15),
                QLabel("Loaded:"), self.ticker_list_widget,
                CustomButton("remove_stock_btn", "top_btns", "indv", self, text="Remove Ticker", height=15),
                self.type_dropdown, self.res_dropdown
            ]
        )

        # Key label for html string
        self.keys_label = QLabel("")
        self.keys_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        add_to_layout(top_layout, [btn_layout, self.keys_label])

        # Graph container
        self.graph_container = QVBoxLayout()
        self.graph = StockGraph(self)
        self.graph_container.addWidget(self.graph.ax.vb.win)

        # Updater visuals
        update_layout = QHBoxLayout()
        self.update_label = QLabel(); self.update_progress = QProgressBar()
        self.update_progress.setMinimumHeight(10)
        self.update_progress.setTextVisible(False)

        add_to_layout(update_layout, [self.update_label, self.update_progress])
        self.updater = UpdateManager(self.update_label, self.update_progress)

        # Add top frame and graph container to center layout
        add_to_layout(center_layout, [top_layout, self.graph_container, update_layout], size_ratios=[1,15,2])
        return center_frame

    # Initialize the right sidebar with profile, prediction settings, and results
    def build_right_frame(self) -> QFrame:
        # Main frame styling
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0 ,0 ,0 ,0)

        # Profile frame styling
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: none;")
        profile_frame_layout = QVBoxLayout(profile_frame)
        profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create profile image icon & logging label and add to profile frame
        desc = "Click to log in" if not self.logged_in else "Click to switch to profile overview window"
        add_to_layout(
            profile_frame_layout, alignment=Qt.AlignmentFlag.AlignCenter,
            items=[create_circle_label(self, diameter=120, desc=desc, border=self.logged_in), self.status_label]
        )

        # Prediction settings frame styling (pd_set = prediction_settings)
        self.pd_set_frame = QFrame()
        self.pd_set_frame.setStyleSheet("border: 1px solid black; font-size: 16px; font-family: Calibri")
        pd_set_layout = QVBoxLayout(self.pd_set_frame); pd_set_layout.setSpacing(20)
        pd_set_layout.setContentsMargins(3 ,3 ,3 ,3)

        pd_label = QLabel("Prediction settings:")
        pd_label.setStyleSheet("border: none; font-weight: bold")

        # Ticker input widget
        self.ticker_pd_input = QLineEdit(); self.ticker_pd_input.setFixedHeight(30)
        self.ticker_pd_input.setPlaceholderText("Ticker symbol...")
        self.ticker_pd_input.setStyleSheet("border: none; border-bottom: 2px solid #999")

        # Prediction type button selection
        pd_type_layout = QHBoxLayout(); pd_type_layout.setSpacing(10)
        add_to_layout(pd_type_layout,
            items=[
                CustomButton("1d", "pd_type_btns", "grp", self, text="day", width=75, height=15),
                CustomButton("1h", "pd_type_btns", "grp", self, text="hour", width=75, height=15),
            ]
        )

        # Confirmation and remove prediction buttons
        confirmations_layout = QHBoxLayout(); confirmations_layout.setSpacing(50)
        add_to_layout(confirmations_layout,
            items=[
                CustomButton("remove_pd_btn", "confirmation_btns", "indv", self, img=abs_file("delete.png"),
                             width=70, height=70, desc="Click to remove prediction from graph"),
                CustomButton("predict_btn", "confirmation_btns", "indv", self, img=abs_file("confirm.png"),
                             width=70, height=70, desc="Click to start prediction."),
            ]
        )

        # Add all prediction setting layouts to prediction settings container
        add_to_layout(
            pd_set_layout, stretches=[-1],
            items=[pd_label, self.ticker_pd_input, pd_type_layout, create_slider_layout(self), confirmations_layout]
        )

        # Prediction result label
        self.pd_result_label = QLabel()
        self.pd_result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pd_result_label.setWordWrap(True)
        self.pd_result_label.setStyleSheet("border: 1px solid black; font-size: 18px; font-family: Calibri")

        # Add profile, prediction settings, and result frames to right frame
        add_to_layout(right_layout, [profile_frame, self.pd_set_frame, self.pd_result_label], size_ratios=[1,10,10])
        return right_frame

    # Helper function to rebuild a select main frame
    def rebuild_frame(self, frame_pos: str) -> None:
        # Get old values and delete the old frame object
        old, stretch = self.main_frames.get(frame_pos)
        index = self.main_layout.indexOf(old)
        self.main_layout.removeWidget(old); old.setParent(None); old.deleteLater()

        # Create new frame and insert back into the correct place
        new = getattr(self, f"build_{frame_pos}_frame")()
        self.main_frames.update({frame_pos: [new, stretch]})
        self.main_layout.insertWidget(index, new, stretch)

    # Helper function to rebuild the graph
    def rebuild_graph(self):
        self.graph_container.removeWidget(self.graph.ax.vb.win)
        self.graph.rebuild_self()
        self.graph_container.addWidget(self.graph.ax.vb.win)

    # Helper function to get a dict of logged in profile
    def get_profile_data(self) -> dict:
        if self.logged_in: return self.logged_profile.get_full_data()
        else: return {}

    # Helper function to add a stock to the graph
    def add_to_graph(self):
        # Get ticker input
        ticker = self.ticker_input.text().strip().upper()
        if ticker == "": return

        # Add it to the graph
        status = self.graph.add_ticker(ticker)
        if status == "No data or invalid ticker":
            QMessageBox.critical(self, "Error", "Invalid ticker")
            return

        self.ticker_input.setText("")

    # Helper function to remove a stock from the graph
    def remove_from_graph(self):
        ticker = self.ticker_list_widget.currentText().strip()
        self.graph.remove_ticker(ticker)

    # Helper function to switch between candlestick and line graph types
    def switch_graph_type(self):
        self.graph.switch_graph_type()

    # Helper function to switch between different time intervals on the graph
    def switch_graph_res(self):
        self.graph.switch_graph_resolution(self.res_dropdown.currentText())

    # Helper function on profile label click
    def label_click(self) -> None:
        # If logged in, open new profile window
        if self.logged_in:
            self.hide()
            profile_window = ProfileWindow(self, self.logged_profile)
            profile_window.exec()
        # If not logged in, display login prompt
        else: self.login_window()

    # Helper function to show login prompt if not logged in
    def login_window(self) -> None:
        # Get username and password input
        username, ok = QInputDialog.getText(self, "Login", "Enter Username:")
        if not ok: return

        password, ok = QInputDialog.getText(self, "Login", f"Enter Password:", QLineEdit.EchoMode.Password)
        if not ok: return

        # Ensure username & password meet length requirements and don't contain illegal characters
        if (not all(6 <= len(word) <= 64 for word in [username, password]) or
            not all(char for char in username if char.isalnum() or char in ["-", "_"])):
            QMessageBox.critical(self, "Error", "Username, password is too short or contains an illegal character.")
            return

        result = self.data_manager.get_profile(username, password)
        # Set status to logged in upon success, and rebuild right frame to update with user information
        if isinstance(result, Profile):
            self.logged_in = True; self.logged_profile = result
            self.rebuild_frame("right")
            self.status_label.setText(f"Status: Logged In as {username}")
            QMessageBox.information(self, "Success", f"Welcome, {username}!")

        # If validation in DataManager class finds no profile, prompt user to create a new profile
        elif result == "Non-existent profile":
            reply = QMessageBox.question(
                self, "Profile Not Found", "Profile does not exist. Would you like to create a new one?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            # If they wish to proceed, create a new profile with details previously entered
            if reply == QMessageBox.StandardButton.Yes:
                result = self.data_manager.create_profile(username, password)
                if result == "Profile created":
                    QMessageBox.information(self, "Success", "New profile created! Please log in again.")
                else: QMessageBox.critical(self, "Creation Error", result)

                self.status_label.setText("Status: New login created")
                return

            self.status_label.setText("Status: Login Failed.")

        # If wrong password was entered, return and display to user error
        elif result == "Incorrect password":
            self.status_label.setText("Status: Incorrect Password.")
            QMessageBox.critical(self, "Login Failed", "Incorrect password.")

        # Catch any other errors and display them
        else: QMessageBox.critical(self, "Error", f"An error occurred: {result}")

    # Helper function to start prediction
    def predict(self) -> None:
        # Get ticker and interval
        ticker = self.ticker_pd_input.text().upper()
        if not validate_ticker(ticker):
            QMessageBox.critical(self, "Error", "Invalid ticker")
            return

        interval = next((btn.name for btn in self.btns["pd_type_btns"] if btn.isChecked()), None)
        for btn in self.btns["pd_type_btns"]: btn.reset()

        # Disable frame for inputs while prediction is being processed
        self.pd_set_frame.setEnabled(False)
        self.pd_result_label.setText("Processing...")

        # Create an instance of the training worker for selected settings
        self.thread = TrainingWorker(ticker, interval)
        self.thread.training_finished.connect(
            lambda res, t=ticker, i=interval: self.prediction_success(ticker, interval, res)
        )
        self.thread.training_error.connect(self.prediction_fail)

        # Ensure the thread closes properly on finish
        self.thread.training_finished.connect(self.thread.quit)
        self.thread.training_finished.connect(self.thread.deleteLater)

        self.thread.start()

    # Helper function to display completed prediction
    def prediction_success(self, ticker: str, interval: str, forecast_results: dict) -> None:
        # Re-enable setting frame
        self.pd_set_frame.setEnabled(True)
        self.ticker_pd_input.setText("")

        # Calculate a threshold for confidence needed to display to user
        risk_level = self.risk_slider.value()
        threshold = 0.5 + 0.35 * np.exp(-0.4 * (risk_level - 1))

        # Build an array of text to dislpay prediction to user in result frame
        results = []
        for time_key, info in forecast_results.items():
            confidence = info['conf']
            warn = '️⚠️ Low confidence!' if confidence < threshold else ''
            results.append(f"<u>For {time_key}{interval[1]}:</u><b>{warn}</b><br>"
                           f"-> Direction: {info['dir']}<br>"
                           f"-> Price: ${info['price']:.2f}<br>"
                           f"-> Confidence: {confidence:.1%}")

        # Show all results to user
        self.res_dropdown.setCurrentText(f"1{interval[0]}")
        self.graph.add_future(ticker, interval, forecast_results)
        self.pd_result_label.setText("<br>".join(results))

    # Helper function on prediction fail
    def prediction_fail(self, error):
        self.pd_set_frame.setEnabled(True)
        self.pd_result_label.setText(f"Prediction Failed: {error}")

    # Ensure script terminates properly on window closure
    def closeEvent(self, event) -> None: event.accept()

############################################################################

