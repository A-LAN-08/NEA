

import sys
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QLabel, QPushButton, QFrame, QDialog, QLineEdit
)
from PyQt5.uic.Compiler.qtproxies import QtWidgets


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Layout Example")
        self.setGeometry(100, 100, 1500, 900)
        self.btns = {}

        ##### --- Main alignment --- #####
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        ##### --- Left area --- #####
        left_frame = QFrame()
        left_frame.setStyleSheet("border: 1px solid black;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(0)

        left_btn_heights = 100
        self.btns["left_btns"] = []

        mouse_btn = self.make_toolbar_btn("mouse_tool", "left_btns", "img_src/mouse_icon_scaled.png", height=left_btn_heights)
        line_tool_btn = self.make_toolbar_btn("line_tool", "left_btns", "img_src/line_icon_scaled.png", height=left_btn_heights)
        notes_tool_btn = self.make_toolbar_btn("notes_tool", "left_btns", "img_src/notes_icon_scaled.png", height=left_btn_heights)

        left_layout.addWidget(mouse_btn)
        left_layout.addWidget(line_tool_btn)
        left_layout.addWidget(notes_tool_btn)
        left_layout.addStretch()

        ##### --- Center area --- #####
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)

        # Top bar
        top_frame = QFrame()
        top_frame.setStyleSheet("border: 1px solid black")
        top_layout = QHBoxLayout(top_frame)
        top_layout.setAlignment(Qt.AlignHCenter)
        top_layout.setContentsMargins(0,0,0,0)
        top_layout.setSpacing(0)

        top_btn_widths = 100
        self.btns["top_btns"] = []

        graph_type_icon = QPushButton()
        graph_type_icon.setCheckable(True)
        graph_type_icon.setFixedWidth(top_btn_widths)
        graph_type_icon.setStyleSheet(f"""
        QPushButton {{background-image: url('img_src/candlestick_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}
        QPushButton:checked {{background-image: url('img_src/line_graph_icon_scaled.png'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:checked:hover {{background-color: #adadad}}        """)
        graph_type_icon.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        graph_type_icon.clicked.connect(lambda checked, n="graph_type": self.testfunc(n))

        add_stock_btn = self.make_unique_btn("add_stock_btn", "top_btns", 'img_src/add_stock_icon_scaled.png', width=top_btn_widths)
        remove_stock_btn = self.make_unique_btn("remove_stock_btn", "top_btns", 'img_src/remove_stock_icon_scaled.png', width=top_btn_widths)
        clear_graph_btn = self.make_unique_btn("clear_graph_btn", "top_btns", 'img_src/clear_graph_icon_scaled.png', width=top_btn_widths)

        save_graph_btn = self.make_unique_btn("save_graph_btn", "top_btns", "img_src/save_graph_icon.png", width=top_btn_widths)

        top_layout.addWidget(graph_type_icon)
        top_layout.addWidget(add_stock_btn)
        top_layout.addWidget(remove_stock_btn)
        top_layout.addWidget(clear_graph_btn)
        top_layout.addStretch()
        top_layout.addWidget(save_graph_btn)

        # Graph area
        graph_frame = self.coloured_frame("transparent")
        graph_label = QLabel("Graph Area")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_frame.layout().addWidget(graph_label)

        center_layout.addWidget(top_frame, 1)
        center_layout.addWidget(graph_frame, 10)

        ##### --- Right sidebar --- #####
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)

        # Profile screen
        profile_frame = QWidget()
        profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame)
        profile_frame_layout.setAlignment(Qt.AlignCenter)

        circle_label = QLabel()
        pixmap = QPixmap("img_src/person_icon.jpg")
        circle_pixmap = self.circle_bitmap(pixmap, 120)
        circle_label.setPixmap(circle_pixmap)
        circle_label.setAlignment(Qt.AlignCenter)

        profile_frame_layout.addWidget(circle_label, alignment=Qt.AlignCenter)

        # Prediction settings frame
        prediction_settings_frame = self.coloured_frame("transparent")

        # Prediction result
        prediction_result_frame = self.coloured_frame("transparent")
        prediction_result_label = QLabel("Prediction result")
        prediction_result_label.setAlignment(Qt.AlignCenter)
        prediction_result_frame.layout().addWidget(prediction_result_label)


        right_layout.addWidget(profile_frame, 1)
        right_layout.addWidget(prediction_settings_frame, 10)
        right_layout.addWidget(prediction_result_frame, 10)

        ##### --- Add to layout --- #####
        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(center_frame, 15)
        main_layout.addWidget(right_frame, 3)

    def testfunc(self, btn):
        print("testfunc", btn.name)
        if btn.name == "save_graph_btn":
            self.show_popup(btn)

    def save_graph(self, input_box):
        self.accept()
        msg = QLabel("Saved.", self)
        msg.setWindowFlags(Qt.Tooltip)
        msg.show()
        QTimer.singleShot(2000, msg.close)
        print(f"Saved. {input_box.text()}")

    def show_popup(self, btn):
        popup = QDialog(self)
        popup.setWindowTitle(btn.name)
        popup.setModal(True)
        popup.setFixedSize(200, 100)
        btn_pos = btn.mapToGlobal(btn.rect().bottomLeft())
        popup.move(btn_pos.x()-50, btn_pos.y())

        layout = QVBoxLayout()

        label = QLabel("Enter the name to save the graph as.")
        input_box = QLineEdit()
        input_box.setPlaceholderText("Name...")
        input_box.returnPressed.connect(lambda i=input_box: self.save_graph(i))

        layout.addWidget(label)
        layout.addWidget(input_box)
        layout.addStretch()
        popup.setLayout(layout)

        popup.exec_()

    def make_unique_btn(self, name, group, img, width = None, height = None):
        btn = QPushButton()
        if height and width:
            btn.setFixedSize(width, height)
        elif height and not width:
            btn.setFixedHeight(height)
        elif width and not height:
            btn.setFixedWidth(width)
        btn.img = img
        btn.name = name
        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}}
        QPushButton:pressed {{background-color: #858585}}        """)
        btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        btn.clicked.connect(lambda checked, b=btn: self.testfunc(b))
        self.btns[group].append(btn)
        return btn

    def make_toolbar_btn(self, name, group, img, width = None, height = None):
        btn = QPushButton()
        btn.setCheckable(True)
        if height and width:
            btn.setFixedSize(width, height)
        elif height and not width:
            btn.setFixedHeight(height)
        elif width and not height:
            btn.setFixedWidth(width)
        btn.img = img
        btn.name = name
        btn.group = group
        btn.setStyleSheet(f"""
        QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
        QPushButton:hover {{background-color: #adadad}} 
        """)

        btn.clicked.connect(lambda checked, b=btn: self.handle_toolbar_btn_click(b))
        self.btns[group].append(btn)
        return btn

    def handle_toolbar_btn_click(self, clicked_btn):
        for btn in self.btns[clicked_btn.group]:
            if btn == clicked_btn:
                btn.setStyleSheet(f"""QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #8a8a8a}}""")
                self.testfunc(btn)
            else:
                btn.setChecked(False)
                btn.setStyleSheet(f"""QPushButton {{background-image: url('{btn.img}'); background-repeat: no-repeat; background-position: center; background-color: #e3e3e3}}
                                      QPushButton:hover {{background-color: #adadad}} """)

    def coloured_frame(self, colour, min_height=None):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setAutoFillBackground(True)
        palette = frame.palette()
        palette.setColor(QPalette.Window, QColor(colour))
        frame.setPalette(palette)
        if min_height:
            frame.setMinimumHeight(min_height)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        return frame

    def circle_bitmap(self, pixmap, diameter):
        pixmap = pixmap.scaled(diameter, diameter, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        mask = QPixmap(diameter, diameter)
        mask.fill(Qt.transparent)

        painter = QPainter(mask)
        path = QPainterPath()
        path.addEllipse(0, 0, diameter, diameter)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return mask

    def closeEvent(self, event) -> None:
        event.accept()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
