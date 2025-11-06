
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.scale_img("img_ogs/line_graph_icon.png", "img_src/line_graph_icon_scaled.png", 100, 80)
        self.scale_img("img_ogs/save_graph_icon.png", "img_src/save_graph_icon.png", 80, 80)

    def scale_img(self, old_path, new_path, x, y) -> None:
        img = QPixmap(old_path).scaled(x, y)
        img.save(new_path)
        print("done.")



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())


