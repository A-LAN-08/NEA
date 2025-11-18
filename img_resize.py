
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.scale_img("img_ogs/line_graph_icon.png", "img_src/line_graph_icon_scaled.png", 100, 80)
        # self.scale_img("img_ogs/save_graph_icon.png", "img_src/save_graph_icon.png", 80, 80)
        a=70
        self.scale_img("img_ogs/reroll.png", "img_src/reroll_icon_scaled.png", a, a)
        self.scale_img("img_ogs/confirm.png", "img_src/confirm_icon_scaled.png", a, a)


    def scale_img(self, old_path, new_path, x, y) -> None:
        img = QPixmap(old_path).scaled(x, y)
        img.save(new_path)
        print("done.")



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())


