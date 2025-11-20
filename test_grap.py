

import sys
# import os
# from statistics import linear_regression

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath#, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QLabel, QPushButton, QFrame, QDialog, QLineEdit, QSlider
)
# from PyQt5.uic.Compiler.qtproxies import QtWidgets


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Layout Example")
        self.setGeometry(100, 100, 1500, 900)
        self.btns = {"left_btns": [], "top_btns": [], "prediction_type_btns": [], "time_period_btns": [], "confirmation_btns": []}
        self.colours = {"Default": "#e3e3e3", "Hover": "#adadad", "Clicked": "#858585"}

        central = QWidget(); self.setCentralWidget(central)

        main_layout = QHBoxLayout(); central.setLayout(main_layout)
        left_frame = self.build_left_frame(); center_frame = self.build_centre_frame(); right_frame = self.build_right_frame()
        main_layout.addWidget(left_frame, 1); main_layout.addWidget(center_frame, 15); main_layout.addWidget(right_frame, 3)

    def build_left_frame(self) -> QFrame:
        ##### --- Left area --- #####
        left_frame = QFrame(); left_frame.setStyleSheet("border: 1px solid black;")
        left_layout = QVBoxLayout(left_frame); left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(0)

        mouse_btn = self.make_img_grp_btn("mouse_tool", "left_btns", "img_src/mouse_icon_scaled.png", height=100)
        line_tool_btn = self.make_img_grp_btn("line_tool", "left_btns", "img_src/line_icon_scaled.png", height=100)
        notes_tool_btn = self.make_img_grp_btn("notes_tool", "left_btns", "img_src/notes_icon_scaled.png", height=100)

        left_layout.addWidget(mouse_btn); left_layout.addWidget(line_tool_btn); left_layout.addWidget(notes_tool_btn)
        left_layout.addStretch()

        return left_frame

    def build_centre_frame(self) -> QFrame:
        ##### --- Center area --- #####
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame)

        ## Top bar
        top_frame = QFrame(); top_frame.setStyleSheet("border: 1px solid black")
        top_layout = QHBoxLayout(top_frame); top_layout.setAlignment(Qt.AlignHCenter); top_layout.setContentsMargins(0,0,0,0); top_layout.setSpacing(0)

        graph_type_btn = QPushButton(); graph_type_btn.setCheckable(True); graph_type_btn.setFixedWidth(100)
        graph_type_btn.name = "graph_type_btn"; graph_type_btn.group = "top_btns"
        graph_type_btn.setStyleSheet(f"""
        QPushButton {{background-image: url('img_src/candlestick_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: {self.colours['Default']}}}
        QPushButton:hover {{background-color: {self.colours['Hover']}}}
        QPushButton:checked {{background-image: url('img_src/line_graph_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: {self.colours['Default']}}}
        QPushButton:checked:hover {{background-color: {self.colours['Hover']}}}        """)
        graph_type_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        graph_type_btn.clicked.connect(lambda checked: self.testfunc(graph_type_btn))

        add_stock_btn = self.make_indv_btn("add_stock_btn", "top_btns", 'img_src/add_stock_icon_scaled.png', width=100)
        remove_stock_btn = self.make_indv_btn("remove_stock_btn", "top_btns", 'img_src/remove_stock_icon_scaled.png', width=100)
        clear_graph_btn = self.make_indv_btn("clear_graph_btn", "top_btns", 'img_src/clear_graph_icon_scaled.png', width=100)
        save_graph_btn = self.make_indv_btn("save_graph_btn", "top_btns", "img_src/save_graph_icon.png", width=100)

        top_layout.addWidget(graph_type_btn); top_layout.addWidget(add_stock_btn); top_layout.addWidget(remove_stock_btn); top_layout.addWidget(clear_graph_btn)
        top_layout.addStretch(); top_layout.addWidget(save_graph_btn)

        ## Graph area
        graph_frame = self.coloured_frame("transparent")
        graph_label = QLabel("Graph Area")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_frame.layout().addWidget(graph_label)

        center_layout.addWidget(top_frame, 1)
        center_layout.addWidget(graph_frame, 10)

        return center_frame

    def build_right_frame(self) -> QFrame:

        ##### --- Right sidebar --- #####
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame)

        ## Profile screen
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame); profile_frame_layout.setAlignment(Qt.AlignCenter)

        circle_label = QLabel()
        circle_label.setPixmap(self.circle_bitmap(QPixmap("img_src/person_icon.jpg"), 120))
        circle_label.setAlignment(Qt.AlignCenter)

        profile_frame_layout.addWidget(circle_label, alignment=Qt.AlignCenter)

        ## Prediction settings frame   (pd_set = prediction settings
        pd_set_frame = QFrame(); pd_set_frame.setStyleSheet("border: 1px solid black")
        pd_set_layout = QVBoxLayout(pd_set_frame); pd_set_layout.setContentsMargins(3,3,3,3); pd_set_layout.setSpacing(20)

        # Ticker input
        ticker_symbol_inbox = QLineEdit(); ticker_symbol_inbox.setPlaceholderText("Ticker symbol...")
        ticker_symbol_inbox.setStyleSheet("font-size: 16px; font-family: Aller Display"); ticker_symbol_inbox.setFixedHeight(30)

        # Type of prediction
        prediction_type_layout = QHBoxLayout(); prediction_type_layout.setSpacing(10)

        lin_reg_btn = self.make_text_grp_btn("linear_regression_btn", "prediction_type_btns", "Linear Reg", width=75, height=30)  # lin_reg = linear regression
        random_forrest_btn = self.make_text_grp_btn("random_forrest_btn", "prediction_type_btns", "Random Forrest", width=75, height=30)
        ri_btn = self.make_text_grp_btn("ri_btn", "prediction_type_btns", "Reinforcement Learning", width=75, height=30)  # ri = reinforcement learning

        prediction_type_layout.addWidget(lin_reg_btn); prediction_type_layout.addWidget(random_forrest_btn); prediction_type_layout.addWidget(ri_btn)

        # Risk slider
        risk_layout = QVBoxLayout(); risk_layout.setContentsMargins(0,0,0,0); risk_layout.setSpacing(0)

        risk_slider = QSlider(Qt.Horizontal); risk_slider.setStyleSheet("""QSlider {border: none}""")
        risk_slider.setTickPosition(QSlider.TicksBelow); risk_slider.setMinimum(1); risk_slider.setMaximum(10); risk_slider.setTickInterval(1); risk_slider.setSingleStep(1)
        def update_value(value): risk_value_label.setText(f"Risk tolerance: {value}{' (Recommended)'if value == 4 else ''}")
        risk_slider.valueChanged.connect(update_value)

        risk_value_label = QLabel("Risk tolerance: 1"); risk_value_label.setAlignment(Qt.AlignCenter); risk_value_label.setStyleSheet("border: none; font-size: 13px; font-family: Aller Display")
        number_layout = QHBoxLayout()
        for i in range(1, 11): nlabel = QLabel(str(i)); nlabel.setAlignment(Qt.AlignCenter); nlabel.setStyleSheet("border: none"); number_layout.addWidget(nlabel)

        risk_layout.addWidget(risk_value_label); risk_layout.addWidget(risk_slider); risk_layout.addLayout(number_layout)

        # Time period
        time_period_layout = QHBoxLayout(); time_period_layout.setSpacing(10)

        day_btn = self.make_text_grp_btn("day_btn", "time_period_btns", "Day", width=75, height=30)
        month_btn = self.make_text_grp_btn("month_btn", "time_period_btns", "Month", width=75, height=30)
        year_btn = self.make_text_grp_btn("year_btn", "time_period_btns", "Year", width=75, height=30)

        time_period_layout.addWidget(day_btn); time_period_layout.addWidget(month_btn); time_period_layout.addWidget(year_btn)

        # Confirmations
        confirmations_layout = QHBoxLayout(); confirmations_layout.setSpacing(50); confirmations_layout.setContentsMargins(20,20,20,20)
        reroll_btn = self.make_indv_btn("reroll_btn", "confirmation_btns", "img_src/reroll_icon_scaled.png", width=70, height=70)
        confirm_pd_btn = self.make_indv_btn("confirm_pd_btn", "confirmation_btns", "img_src/confirm_icon_scaled.png", width=70, height=70)

        confirmations_layout.addWidget(reroll_btn); confirmations_layout.addWidget(confirm_pd_btn)

        pd_set_layout.addWidget(ticker_symbol_inbox); pd_set_layout.addLayout(prediction_type_layout); pd_set_layout.addLayout(risk_layout)
        pd_set_layout.addLayout(time_period_layout); pd_set_layout.addLayout(confirmations_layout); pd_set_layout.addStretch()

        ## Prediction result
        prediction_result_frame = self.coloured_frame("transparent")
        prediction_result_label = QLabel("Prediction result"); prediction_result_label.setAlignment(Qt.AlignCenter)
        prediction_result_frame.layout().addWidget(prediction_result_label)

        right_layout.addWidget(profile_frame, 1); right_layout.addWidget(pd_set_frame, 10); right_layout.addWidget(prediction_result_frame, 10)
        return right_frame

    def testfunc(self, btn: QPushButton):
        print("testfunc", btn.name)
        if btn.name == "save_graph_btn":
            self.show_popup(btn)

    def save_graph(self, input_box):
        print(f"Saved. {input_box.text()}")
        msg = QWidget(self); msg.setWindowFlags(Qt.FramelessWindowHint | Qt.BypassWindowManagerHint); msg.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(msg)
        label = QLabel("Saved."); label.setStyleSheet("background-color: black; color: white; padding: 5px; border-radius: 5px;"); layout.addWidget(label)

        msg.adjustSize(); pos = self.rect().center() - msg.rect().center(); msg.move(pos); msg.show()
        QTimer.singleShot(2000, msg.close)

    def show_popup(self, btn):
        popup = QDialog(self); popup.setWindowTitle(btn.name); popup.setModal(True); popup.setFixedSize(200, 100)
        btn_pos = btn.mapToGlobal(btn.rect().bottomLeft())
        popup.move(btn_pos.x()-50, btn_pos.y())

        layout = QVBoxLayout()
        label = QLabel("Enter the name to save the graph as.")
        input_box = QLineEdit(); input_box.setPlaceholderText("Name...")

        def save_and_close(): self.save_graph(input_box); popup.accept()
        input_box.returnPressed.connect(save_and_close)

        layout.addWidget(label); layout.addWidget(input_box); layout.addStretch()
        popup.setLayout(layout); popup.exec_()

    def make_indv_btn(self, name, group, img, width = None, height = None):
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

        btn.clicked.connect(lambda checked, b=btn: self.testfunc(b))
        self.btns[group].append(btn)
        return btn

    def make_text_grp_btn(self, name, group, text, width = None, height = None):
        btn = QPushButton(); btn.setCheckable(True)
        btn.name = name; btn.group = group; btn.text = text

        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)
        
        btn.setStyleSheet("""
        QPushButton {background-color: #e3e3e3; font-size: 13px; font-family: Aller display}
        QPushButton:hover {background-color: #adadad}""")
        btn.setText(btn.text)

        def handle_text_grp_btn_click(clicked_btn):
            for grp_btn in self.btns[clicked_btn.group]:
                if grp_btn == clicked_btn:
                    grp_btn.setStyleSheet("QPushButton {background-color: #8a8a8a; font-size: 13px; font-family: Aller display}")
                    self.testfunc(grp_btn)
                else:
                    grp_btn.setChecked(False)
                    grp_btn.setStyleSheet("""QPushButton {background-color: #e3e3e3; font-size: 13px; font-family: Aller display}
                                          QPushButton:hover {background-color: #adadad} """)

        btn.clicked.connect(lambda checked: handle_text_grp_btn_click(btn))
        self.btns[group].append(btn)
        return btn

    def make_img_grp_btn(self, name, group, img, width = None, height = None):
        btn = QPushButton(); btn.setCheckable(True)
        btn.name = name; btn.group = group; btn.img = img
        
        if height and width: btn.setFixedSize(width, height)
        elif height and not width: btn.setFixedHeight(height)
        elif width and not height: btn.setFixedWidth(width)

        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}""")

        def handle_img_grp_btn_click(clicked_btn):
            for grp_btn in self.btns[clicked_btn.group]:
                if grp_btn == clicked_btn:
                    grp_btn.setStyleSheet(f"""QPushButton {{background-image: url('{grp_btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #8a8a8a}}""")
                    self.testfunc(grp_btn)
                else:
                    grp_btn.setChecked(False)
                    grp_btn.setStyleSheet(f"""QPushButton {{background-image: url('{grp_btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
                                          QPushButton:hover {{background-color: #adadad}} """)

        btn.clicked.connect(lambda checked: handle_img_grp_btn_click(btn))
        self.btns[group].append(btn)
        return btn

    def coloured_frame(self, colour, min_height=None):    # TEMP FUNCTION
        frame = QFrame(); frame.setFrameShape(QFrame.StyledPanel); frame.setAutoFillBackground(True)
        palette = frame.palette(); palette.setColor(QPalette.Window, QColor(colour))
        frame.setPalette(palette)
        if min_height: frame.setMinimumHeight(min_height)
        layout = QVBoxLayout(frame); layout.setContentsMargins(5, 5, 5, 5)
        return frame

    def circle_bitmap(self, pixmap, diameter):
        pixmap = pixmap.scaled(diameter, diameter, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        mask = QPixmap(diameter, diameter); mask.fill(Qt.transparent)

        painter = QPainter(mask)
        path = QPainterPath(); path.addEllipse(0, 0, diameter, diameter)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap); painter.end()
        return mask

    def closeEvent(self, event) -> None: event.accept()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
