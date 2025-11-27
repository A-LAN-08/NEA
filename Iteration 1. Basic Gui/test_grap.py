

import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QWidget, QLabel, QFrame, QPushButton, QDialog, QLineEdit, QSlider, QMessageBox)


class MainWindow(QMainWindow):
    def __init__(self):
        # Initialize the main window and dictionaries for buttons and colours
        super().__init__()
        self.setWindowTitle("Stock Prediction App")
        self.setGeometry(100, 100, 1500, 900)
        self.btns = {"left_btns": [], "top_btns": [], "prediction_type_btns": [], "time_period_btns": [], "confirmation_btns": []}
        self.colours = {"Default": "#e3e3e3", "Hover": "#adadad", "Clicked": "#858585"}

        # Set up the main layout with left, center, and right frames
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(); central.setLayout(main_layout)
        left_frame = self.build_left_frame(); center_frame = self.build_center_frame(); right_frame = self.build_right_frame()
        main_layout.addWidget(left_frame, 1); main_layout.addWidget(center_frame, 15); main_layout.addWidget(right_frame, 3)

    def build_left_frame(self) -> QFrame:
        # Initialize the left sidebar with tool buttons
        left_frame = QFrame(); left_frame.setStyleSheet("border: 1px solid black;")
        left_layout = QVBoxLayout(left_frame); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(0)

        # Define tool buttons
        mouse_btn = self.make_img_grp_btn("mouse_tool", "left_btns", "img_src/mouse_icon_scaled.png", height=100)
        line_tool_btn = self.make_img_grp_btn("line_tool", "left_btns", "img_src/line_icon_scaled.png", height=100)
        notes_tool_btn = self.make_img_grp_btn("notes_tool", "left_btns", "img_src/notes_icon_scaled.png", height=100)

        left_layout.addWidget(mouse_btn); left_layout.addWidget(line_tool_btn); left_layout.addWidget(notes_tool_btn); left_layout.addStretch()
        return left_frame

    def build_center_frame(self) -> QFrame:
        # Initialize the center frame with top bar and graph area
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame)

        # Defiine top frame
        top_frame = QFrame(); top_frame.setStyleSheet("border: 1px solid black")
        top_layout = QHBoxLayout(top_frame); top_layout.setAlignment(Qt.AlignHCenter); top_layout.setContentsMargins(0,0,0,0); top_layout.setSpacing(0)

        # Define graph type toggle button
        graph_type_btn = QPushButton(); graph_type_btn.setCheckable(True); graph_type_btn.setFixedWidth(100)
        graph_type_btn.name = "graph_type_btn"; graph_type_btn.group = "top_btns"
        graph_type_btn.setStyleSheet(f"""
        QPushButton {{background-image: url('img_src/candlestick_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: {self.colours['Default']}}}
        QPushButton:hover {{background-color: {self.colours['Hover']}}}
        QPushButton:checked {{background-image: url('img_src/line_graph_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: {self.colours['Default']}}}
        QPushButton:checked:hover {{background-color: {self.colours['Hover']}}}        """)
        graph_type_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        graph_type_btn.clicked.connect(lambda checked: self.testfunc(graph_type_btn))

        # Define graph stock edit buttons
        add_stock_btn = self.make_indv_btn("add_stock_btn", "top_btns", 'img_src/add_stock_icon_scaled.png', width=100)
        remove_stock_btn = self.make_indv_btn("remove_stock_btn", "top_btns", 'img_src/remove_stock_icon_scaled.png', width=100)
        clear_graph_btn = self.make_indv_btn("clear_graph_btn", "top_btns", 'img_src/clear_graph_icon_scaled.png', width=100)
        save_graph_btn = self.make_indv_btn("save_graph_btn", "top_btns", "img_src/save_graph_icon.png", width=100)

        top_layout.addWidget(graph_type_btn); top_layout.addWidget(add_stock_btn); top_layout.addWidget(remove_stock_btn); top_layout.addWidget(clear_graph_btn)
        top_layout.addStretch(); top_layout.addWidget(save_graph_btn)

        # Define graph frame (TBD: to be developed further)
        graph_frame = self.coloured_frame("transparent")
        graph_label = QLabel("Graph Area")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_frame.layout().addWidget(graph_label)

        # Add top frame and graph frame to center layout
        center_layout.addWidget(top_frame, 1); center_layout.addWidget(graph_frame, 10)
        return center_frame

    def build_right_frame(self) -> QFrame:
        # Initialize the right sidebar with profile, prediction settings, and results
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame)

        # Define profile frame
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignCenter)

        # Create profile widget with circular pixmap
        circle_label = QLabel()
        circle_label.setPixmap(self.circle_bitmap(QPixmap("img_src/person_icon.jpg"), 120))
        circle_label.setAlignment(Qt.AlignCenter)

        profile_frame_layout.addWidget(circle_label, alignment=Qt.AlignCenter)

        ## Define prediction settings frame (pd_set = prediction_settings) and widgets within
        self.pd_set_frame = QFrame(); self.pd_set_frame.setStyleSheet("border: 1px solid black")
        pd_set_layout = QVBoxLayout(self.pd_set_frame); pd_set_layout.setContentsMargins(3,3,3,3); pd_set_layout.setSpacing(20)

        # Ticker input widget
        self.ticker_symbol_inbox = QLineEdit(); self.ticker_symbol_inbox.setPlaceholderText("Ticker symbol...")
        self.ticker_symbol_inbox.setStyleSheet("font-size: 16px; font-family: Aller Display"); self.ticker_symbol_inbox.setFixedHeight(30)

        # Type of prediction selection widgets
        prediction_type_layout = QHBoxLayout(); prediction_type_layout.setSpacing(10)

        lin_reg_btn = self.make_text_grp_btn("linear_regression_btn", "prediction_type_btns", "Linear Reg", width=75, height=30)  # lin_reg = linear regression
        random_forrest_btn = self.make_text_grp_btn("random_forrest_btn", "prediction_type_btns", "Random Forrest", width=75, height=30)
        ri_btn = self.make_text_grp_btn("ri_btn", "prediction_type_btns", "Reinforcement Learning", width=75, height=30)  # ri = reinforcement learning

        prediction_type_layout.addWidget(lin_reg_btn); prediction_type_layout.addWidget(random_forrest_btn); prediction_type_layout.addWidget(ri_btn)

        # Risk slider widget
        risk_layout = QVBoxLayout(); risk_layout.setContentsMargins(0,0,0,0); risk_layout.setSpacing(0)

        self.risk_slider = QSlider(Qt.Horizontal); self.risk_slider.setStyleSheet("""QSlider {border: none}"""); self.risk_slider.setTickPosition(QSlider.TicksBelow)
        self.risk_slider.setMinimum(1); self.risk_slider.setMaximum(10); self.risk_slider.setTickInterval(1); self.risk_slider.setSingleStep(1)
        self.risk_slider.valueChanged.connect(lambda v: risk_value_label.setText(f"Risk tolerance: {v}{' (Recommended)'if v == 4 else ''}"))

        # Risk slider labels
        risk_value_label = QLabel("Risk tolerance: 1"); risk_value_label.setAlignment(Qt.AlignCenter); risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
        number_layout = QHBoxLayout()
        for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

        risk_layout.addWidget(risk_value_label); risk_layout.addWidget(self.risk_slider); risk_layout.addLayout(number_layout)

        # Time period selection widgets
        time_period_layout = QHBoxLayout(); time_period_layout.setSpacing(10)

        day_btn = self.make_text_grp_btn("day_btn", "time_period_btns", "Day", width=75, height=30)
        month_btn = self.make_text_grp_btn("month_btn", "time_period_btns", "Month", width=75, height=30)
        year_btn = self.make_text_grp_btn("year_btn", "time_period_btns", "Year", width=75, height=30)

        time_period_layout.addWidget(day_btn); time_period_layout.addWidget(month_btn); time_period_layout.addWidget(year_btn)

        # Confirmation and redo widgets
        confirmations_layout = QHBoxLayout(); confirmations_layout.setSpacing(50); confirmations_layout.setContentsMargins(20,20,20,20)
        reroll_btn = self.make_indv_btn("reroll_btn", "confirmation_btns", "img_src/reroll_icon_scaled.png", width=70, height=70)
        confirm_pd_btn = self.make_indv_btn("confirm_pd_btn", "confirmation_btns", "img_src/confirm_icon_scaled.png", width=70, height=70)

        confirmations_layout.addWidget(reroll_btn); confirmations_layout.addWidget(confirm_pd_btn)

        # Add all prediction setting widgets to prediction settings layout
        pd_set_layout.addWidget(self.ticker_symbol_inbox); pd_set_layout.addLayout(prediction_type_layout); pd_set_layout.addLayout(risk_layout)
        pd_set_layout.addLayout(time_period_layout); pd_set_layout.addLayout(confirmations_layout); pd_set_layout.addStretch()

        # Define prediction result widget (TBD: to be developed further)
        prediction_result_frame = QFrame(); prediction_result_frame.setStyleSheet("border: 1px solid black")
        prediction_result_layout = QVBoxLayout(prediction_result_frame)
        self.prediction_result_label = QLabel("Prediction result"); self.prediction_result_label.setAlignment(Qt.AlignCenter); self.prediction_result_label.setWordWrap(True) 
        self.prediction_result_label.setStyleSheet("border: none")
        prediction_result_layout.addWidget(self.prediction_result_label)

        # Add profile, prediction settings, and result frames to right frame
        right_layout.addWidget(profile_frame, 1); right_layout.addWidget(self.pd_set_frame, 10); right_layout.addWidget(prediction_result_frame, 10)
        return right_frame

    def testfunc(self, btn: QPushButton) -> None:
        # Temporary function to test button click activation
        print("testfunc", btn.name)
        if btn.name == "save_graph_btn":
            self.show_graph_save_popup(btn)
        elif btn.name == "confirm_pd_btn":
            self.start_prediction_simulation()

    def start_prediction_simulation(self) -> None:
        print("Prediction starting...") # DEBUG
        # Find ticker and risk level inputs
        ticker = self.ticker_symbol_inbox.text(); risk_level = self.risk_slider.value()
        # Find selected Prediction Type button        
        selected_prediction_type = next((btn.text for btn in self.btns["prediction_type_btns"] if btn.isChecked()), None)
        # Find selected Time Period button
        selected_time_period = next((btn.text for btn in self.btns["time_period_btns"] if btn.isChecked()), None)
        
        # Validation to make sure all input fields are filled
        if not all([ticker, selected_prediction_type, selected_time_period]):
            QMessageBox.warning(self, "Input Error", "Please fill in all prediction settings before confirming.")
            return

        self.pd_set_frame.setEnabled(False)
        self.prediction_result_label.setText("Processing... (10s)")

        def finish_prediction_simulation():
            print("Prediction finished.")  # DEBUG
            # Re-enable prediction settings and show results (TBD: to be developed further)
            self.pd_set_frame.setEnabled(True)
            self.prediction_result_label.setText(f"""
Completed Prediction. . .                       
--- INPUTS RECEIVED ---
Ticker: {ticker}
Prediction Type: {selected_prediction_type}
Risk Level: {risk_level}
Time Period: {selected_time_period}
-----------------------""")  # Temp text to show completion
            # Popup message box to show success
            QMessageBox.information(self, "Prediction Status", "Successful")

        QTimer.singleShot(10000, finish_prediction_simulation)

    def save_graph(self, input_box) -> None:
        # Function to save the state of the graph when button pressed (TBD: to be developed further)
        print(f"Saved. {input_box.text()}")
        msg = QWidget(self); msg.setWindowFlags(Qt.FramelessWindowHint | Qt.BypassWindowManagerHint); msg.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(msg)
        label = QLabel("Saved."); label.setStyleSheet("background-color: black; color: white; padding: 5px; border-radius: 5px;"); layout.addWidget(label)

        msg.adjustSize(); pos = self.rect().center() - msg.rect().center(); msg.move(pos); msg.show()
        QTimer.singleShot(2000, msg.close)

    def show_graph_save_popup(self, btn) -> None:
        # Function to show popup for saving graph (TBD: to be developed further)
        # Creates popup dialog and positions it below the button
        popup = QDialog(self); popup.setWindowTitle(btn.name); popup.setModal(True); popup.setFixedSize(200, 100)
        btn_pos = btn.mapToGlobal(btn.rect().bottomLeft())
        popup.move(btn_pos.x()-50, btn_pos.y())

        # Take input from popup
        layout = QVBoxLayout()
        label = QLabel("Enter the name to save the graph as.")
        input_box = QLineEdit(); input_box.setPlaceholderText("Name...")

        # On enter pressed, save graph and close popup
        def save_and_close(): self.save_graph(input_box); popup.accept()
        input_box.returnPressed.connect(save_and_close)

        layout.addWidget(label); layout.addWidget(input_box); layout.addStretch()
        popup.setLayout(layout); popup.exec_()

    def make_indv_btn(self, name, group, img, width = None, height = None) -> QPushButton:
        # Creates an independent button calls a function every time it is clicked
        # Setup button properties
        btn = QPushButton()
        btn.img = img; btn.name = name; btn.group = group

        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)
            
        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}
        QPushButton:pressed {{background-color: #858585}}        """)
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Call testfunc on click
        btn.clicked.connect(lambda checked, b=btn: self.testfunc(b))
        self.btns[group].append(btn); return btn

    def make_text_grp_btn(self, name, group, text, width = None, height = None) -> QPushButton:
        # Create a button part of group with text label as its filler
        # Setup button properties
        btn = QPushButton(); btn.setCheckable(True)
        btn.name = name; btn.group = group; btn.text = text

        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)
        
        btn.setStyleSheet("""
        QPushButton {background-color: #e3e3e3; font-size: 13px; font-family: Aller display}
        QPushButton:hover {background-color: #adadad}""")
        btn.setText(btn.text)

        # Define how other buttons in group respond when one is clicked
        def handle_text_grp_btn_click(clicked_btn):
            for grp_btn in self.btns[clicked_btn.group]:
                if grp_btn == clicked_btn:
                    # Set state to clicked style
                    grp_btn.setStyleSheet("QPushButton {background-color: #8a8a8a; font-size: 13px; font-family: Aller display}")
                    self.testfunc(grp_btn)
                else:
                    # Set state to unclicked style
                    grp_btn.setChecked(False)
                    grp_btn.setStyleSheet("""QPushButton {background-color: #e3e3e3; font-size: 13px; font-family: Aller display}
                                          QPushButton:hover {background-color: #adadad} """)

        # Call click handler on click
        btn.clicked.connect(lambda checked: handle_text_grp_btn_click(btn))
        self.btns[group].append(btn); return btn

    def make_img_grp_btn(self, name, group, img, width = None, height = None) -> QPushButton:
        # Create a button part of group with image as its filler
        # Setup button properties
        btn = QPushButton(); btn.setCheckable(True)
        btn.name = name; btn.group = group; btn.img = img
        
        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)

        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}""")

        # Define how other buttons in group respond when one is clicked
        def handle_img_grp_btn_click(clicked_btn):
            for grp_btn in self.btns[clicked_btn.group]:
                if grp_btn == clicked_btn:
                    # Set state to clicked style
                    grp_btn.setStyleSheet(f"""QPushButton {{background-image: url('{grp_btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #8a8a8a}}""")
                    self.testfunc(grp_btn)
                else:
                    # Set state to unclicked style
                    grp_btn.setChecked(False)
                    grp_btn.setStyleSheet(f"""QPushButton {{background-image: url('{grp_btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
                                          QPushButton:hover {{background-color: #adadad}} """)

        # Call click handler on click
        btn.clicked.connect(lambda checked: handle_img_grp_btn_click(btn))
        self.btns[group].append(btn); return btn

    def coloured_frame(self, colour, min_height=None) -> QFrame:    # TEMP FUNCTION
        # Create a frame with a coloured border
        frame = QFrame(); frame.setFrameShape(QFrame.StyledPanel); frame.setAutoFillBackground(True)
        palette = frame.palette(); palette.setColor(QPalette.Window, QColor(colour))
        frame.setPalette(palette)
        if min_height: frame.setMinimumHeight(min_height)
        layout = QVBoxLayout(frame); layout.setContentsMargins(5, 5, 5, 5)
        return frame

    def circle_bitmap(self, pixmap, diameter) -> QPixmap:
        # Create a circular pixmap to use as a filler area
        pixmap = pixmap.scaled(diameter, diameter, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        mask = QPixmap(diameter, diameter); mask.fill(Qt.transparent)

        painter = QPainter(mask)
        path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap); painter.end()
        return mask

    def closeEvent(self, event) -> None: event.accept()

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
