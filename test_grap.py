import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette, QPainter, QPixmap, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Layout Example")
        self.setGeometry(100, 100, 1500, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_bar = self.coloured_frame("lightgray")
        left_layout = QVBoxLayout(left_bar)
        left_layout.addStretch()

        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)

        top_bar = self.coloured_frame("whitesmoke")
        top_layout = QHBoxLayout(top_bar)
        top_layout.addStretch()

        graph_frame = self.coloured_frame("white")
        graph_layout = QVBoxLayout(graph_frame)
        graph_label = QLabel("Graph Area")
        graph_label.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(graph_label)

        center_layout.addWidget(top_bar, 1)
        center_layout.addWidget(graph_frame, 10)

        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)

        profile_bar = QWidget()
        profile_bar.setStyleSheet("background-color: lightgray;")
        right_bar_profile_layout = QVBoxLayout(profile_bar)
        right_bar_profile_layout.setAlignment(Qt.AlignCenter)

        circle_label = QLabel()
        pixmap = QPixmap("img.jpg")
        if pixmap.isNull():
            pixmap = QPixmap(200, 200)
            pixmap.fill(Qt.gray)
        circle_pixmap = self.circle_bitmap(pixmap, 120)
        circle_label.setPixmap(circle_pixmap)
        circle_label.setAlignment(Qt.AlignCenter)

        right_bar_profile_layout.addWidget(circle_label, alignment=Qt.AlignCenter)

        right_bar_prediction_settings = self.coloured_frame("red")

        right_bar_prediction = self.coloured_frame("lightgray")
        right_bar_prediction_layout = QVBoxLayout(right_bar_prediction)
        right_bar_label = QLabel("Prediction result")
        right_bar_label.setAlignment(Qt.AlignCenter)
        right_bar_prediction_layout.addWidget(right_bar_label)

        right_layout.addWidget(profile_bar, 1)
        right_layout.addWidget(right_bar_prediction_settings, 10)
        right_layout.addWidget(right_bar_prediction, 10)

        main_layout.addWidget(left_bar, 1)
        main_layout.addWidget(center_frame, 15)
        main_layout.addWidget(right_frame, 3)

    def coloured_frame(self, colour, min_height=None):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setAutoFillBackground(True)
        palette = frame.palette()
        palette.setColor(QPalette.Window, QColor(colour))
        frame.setPalette(palette)
        if min_height:
            frame.setMinimumHeight(min_height)
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

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
