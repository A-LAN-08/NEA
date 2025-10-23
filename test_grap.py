

import sys
import os

from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Layout Example")
        self.setGeometry(100, 100, 1500, 900)

        ##### --- Main alignment --- #####
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        ##### --- Left sidebar --- #####
        left_frame = QFrame()
        left_frame.setStyleSheet("border: 1px solid black;")

        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0,0,0,0)#1,1,1,1)
        left_layout.setSpacing(1)

        left_btn_heights = 100

        mouse_btn = QPushButton()
        self.scale_img("mouse.jpg", "mouse_scaled.jpg", 40, 50)

        mouse_btn.setStyleSheet("""
        QPushButton {background-image: url('mouse_scaled.jpg'); background-repeat: no-repeat; background-position: center}
        QPushButton:hover {background: #34abeb} 
        QPushButton:pressed {background-color: #1338bd} 
        """)
        mouse_btn.setFixedHeight(left_btn_heights)

        mouse_btn.clicked.connect(self.testfunc)

        # line_tool = QPushButton()
        # line_tool.setFixedHeight(left_btn_heights)

        left_layout.addWidget(mouse_btn)
        # left_layout.addWidget(line_tool)
        left_layout.addStretch()

        ##### --- Center bar --- #####
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)

        # Top centre
        top_centre = self.coloured_frame("transparent")
        top_centre_layout = QHBoxLayout(top_centre)
        top_centre_layout.addStretch()

        # Graph area
        graph_frame = self.coloured_frame("transparent")
        graph_label = QLabel("Graph Area")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_frame.layout().addWidget(graph_label)

        center_layout.addWidget(top_centre, 1)
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
        pixmap = QPixmap("person_icon.jpg")
        if pixmap.isNull():
            pixmap = QPixmap(200, 200)
            pixmap.fill(Qt.gray)
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

    def testfunc(self):
        print("testfunc")

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

    def scale_img(self, old_path, new_path, x, y):
        img = QPixmap(old_path).scaled(x,y)
        img.save(new_path)
        created_paths.append(new_path)

    def closeEvent(self, event):
        for created_path in created_paths:
            os.remove(created_path)
        event.accept()



created_paths = []

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
