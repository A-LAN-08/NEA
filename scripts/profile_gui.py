
from __future__ import annotations

# Standard library imports
import io
import json
import os
from typing import TYPE_CHECKING

# External library imports
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QHBoxLayout, QVBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMenu, QMessageBox, QScrollArea, QSlider, QWidget
)

# Initialize matplotlib settings
plt.ioff()

# Custom imports
from scripts.data_management import abs_file, peek_data, validate_ticker
from scripts.profile_management import Profile
from scripts.custom_widgets import add_to_layout, create_circle_label, create_slider_layout, CustomButton
from scripts.config import ICON_DIR

# Type hinting imports
if TYPE_CHECKING:
    from scripts.default_gui import MainWindow

############################################################################

# Class for main profile window
class ProfileWindow(QDialog):
    # Type hints
    search_input: QLineEdit
    risk_slider: QSlider

    def __init__(self, parent_window: MainWindow, profile_obj: Profile):
        # Initialize the main window and dictionaries for button groups
        super().__init__(parent_window)
        self.setWindowTitle(f"Profile Management - {profile_obj.get_username()}")
        self.setGeometry(100, 100, 1500, 900)
        self.parent_window = parent_window
        self.logged_profile = profile_obj
        self.btns = {"profile_btns": [], "Na": []}

        # Switch back to main window when closed
        self.rebuild_parent = False
        self.finished.connect(self.show_parent_on_close)

        # Set up the main layout and save to dict for reframing later
        self.main_layout = QVBoxLayout(self)
        self.main_frames = {"top": [self.build_top_frame(), 1], "bottom": [self.build_bottom_frame(), 2]}
        for frame_info in self.main_frames.values(): self.main_layout.addWidget(frame_info[0], frame_info[1])

    # Initialize the top frame with profile settings and preferences
    def build_top_frame(self) -> QFrame:
        # Main frame styling
        top_frame = QFrame(); top_layout = QHBoxLayout(top_frame); top_layout.setContentsMargins(75, 5, 5, 20)

        # Create profile widget and username label, and add to a container layout
        profile_frame = QWidget(); profile_frame.setStyleSheet("background-color: None;")
        profile_frame_layout = QVBoxLayout(profile_frame)
        profile_frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_to_layout(
            profile_frame_layout, alignment=Qt.AlignmentFlag.AlignCenter,
            items=[create_circle_label(self, clickable=True, diameter=120, desc="Click to select a profile image"),
                   QLabel(self.logged_profile.get_username()) ]
        )

        # Settings frame styling
        setting_frame = QFrame(); setting_frame.setStyleSheet("border: 1px solid black")
        setting_frame.setFixedHeight(200)
        setting_layout = QVBoxLayout(setting_frame)
        setting_layout.setContentsMargins(0,0,0,0); setting_layout.setSpacing(10)

        # Create edit profile buttons
        edit_frame = QFrame(); edit_frame.setStyleSheet("border: 1px solid black")
        edit_layout = QHBoxLayout(edit_frame); edit_layout.setContentsMargins(0,0,0,0)
        edit_layout.setSpacing(0)

        edit_btns = [
            ("logout_btn", abs_file("logout.png"), "Click to logout of profile"),
            ("change_profile_btn", abs_file("change_profile.png"), "Click to change profile"),
            ("export_data_btn", abs_file("export_data.png"), "Click to export saved data"),
            ("import_data_btn", abs_file("import_data.png"), "Click to import data"),
            ("delete_profile_btn", abs_file("delete.png"), "Click to delete profile. WARNING: Irreversible."),
        ]

        # Add widget and layouts to correct outer layouts
        add_to_layout(
            edit_layout,
            [CustomButton(name, "profile_btns", "indv", parent=self, img=img, desc=desc)
             for name, img, desc in edit_btns]
        )
        add_to_layout(setting_layout, [edit_frame, create_slider_layout(self)])
        add_to_layout(top_layout, [profile_frame, setting_frame], stretches=[1,-1])
        return top_frame

    # Initialize bottom frame with saved stocks overviews
    def build_bottom_frame(self) -> QFrame:
        # Main frame styling
        bottom_frame = QFrame(); bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 10, 10, 10); bottom_layout.setSpacing(15)

        ## Left compact frame styling
        left_frame = QFrame()
        left_frame.setFixedWidth(250); left_frame.setStyleSheet("border: 1px solid black")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0,0,0,0); left_layout.setSpacing(3)

        compound_stocks_widget = QScrollArea()
        compound_stocks_widget.setWidgetResizable(True); compound_stocks_widget.setStyleSheet("border: none")
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget); scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create a widget with ticker and percent change for every saved stock
        saved_tickers = self.logged_profile.get_data().get("Saved stocks", [])
        for index, ticker in enumerate(saved_tickers):
            # Find latest day of data
            data = peek_data(ticker, 1)
            if data is None: continue

            # Get price change
            last_price, now_price = data.iloc[0]['Close'], data.iloc[-1]['Close']
            price_change = round(abs((now_price - last_price) / last_price) * 100, 2)

            # Frame for the row styling
            row_frame = QFrame(); row_frame.setFixedHeight(40)
            row_frame.setStyleSheet("border: 1px solid gray; padding: 5px")
            row_layout = QHBoxLayout(row_frame); row_layout.setContentsMargins(10,0,10,0)

            # Create label for ticker
            ticker_label = QLabel(ticker); ticker_label.setStyleSheet("border: none; font-weight: bold")
            percent_label = QLabel(f"{'+' if last_price < now_price else '-'}{price_change}%")
            percent_label.setStyleSheet(f"""border: none; color: {'#008000' if last_price < now_price else '#FF0000'}""")

            # Add widget to the row and set function call on click
            add_to_layout(row_layout, [ticker_label, percent_label], stretches=[1])
            row_frame.setCursor(Qt.CursorShape.PointingHandCursor)
            row_frame.mousePressEvent = lambda event, t=ticker, i=index: self.show_stock_menu(t, i)

            scroll_layout.addWidget(row_frame)
        # Add row container to main left layout
        compound_stocks_widget.setWidget(scroll_widget); left_layout.addWidget(compound_stocks_widget)

        ## Right detailed frame styling
        right_frame = QFrame(); right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create search bar and button to submit search
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for a stock..."); self.search_input.setFixedHeight(40)
        self.search_input.setStyleSheet("border: 1px solid black; padding-left: 10px;")
        self.search_input.returnPressed.connect(self.add_stock)

        confirm_btn = CustomButton(
            "search_confirm_btn", "Na", "indv", parent=self, img=abs_file("confirm.png"), width=60, height=40
        )
        confirm_btn.setProperty("BorderBlank", "true")
        confirm_btn.style().unpolish(confirm_btn); confirm_btn.style().polish(confirm_btn)
        add_to_layout(search_layout, [self.search_input, confirm_btn])

        # Container for rows of detailed info
        scroll_container = QFrame(); scroll_container.setStyleSheet("border: 1px solid black")
        scroll_container_layout = QVBoxLayout(scroll_container)
        scroll_container_layout.setContentsMargins(0, 0, 0, 0)

        detailed_stocks_widget = QScrollArea(); detailed_stocks_widget.setWidgetResizable(True)
        detailed_stocks_widget.setStyleSheet("border: None")
        scroll_widget = QWidget(); scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Iterate through every saved ticker and add detailed row to scroll container
        for index, ticker in enumerate(saved_tickers):
            scroll_layout.addWidget(self.create_stock_card(ticker, index))
        detailed_stocks_widget.setWidget(scroll_widget)
        scroll_container_layout.addWidget(detailed_stocks_widget)

        # Add to layout and return whole frame
        add_to_layout(right_layout, [search_layout, scroll_container])
        add_to_layout(bottom_layout, [left_frame, right_frame])
        return bottom_frame

    # Helper function to create a detailed stock card
    def create_stock_card(self, ticker: str, index: int) -> QFrame | None:
        # Find latest 30 days of data
        df = peek_data(ticker, 30, "1d")
        if df is None: return None

        # Get price change
        last_price, now_price = df.iloc[0]['Close'], df.iloc[-1]['Close']
        price_change = round(abs((now_price - last_price) / last_price) * 100, 2)
        colour = '#008000' if now_price >= last_price else '#FF0000'

        # Frame for the row styling
        card = QFrame(); card.setFixedHeight(120)
        card.setStyleSheet("border: 2px solid black; margin-bottom: 5px; background-color: white;")
        layout = QHBoxLayout(card)

        # Create a graph of latest data
        fig, ax = plt.subplots(figsize=(4, 0.8), dpi=100); ax.margins(x=0, y=0.05)
        ax.plot(df.index, df['Close'], color=colour, linewidth=2)

        # Make graph "Invisible" (Hide all axes/borders)
        ax.axis('off'); fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_visible(False)

        # Save graph to RAM instead of Disk and clean up memory immediately
        buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True)
        buf.seek(0); plt.close(fig)

        # Convert graph image to QPixmap and set it to a label
        chart_area = QLabel(); chart_area.setStyleSheet("border: 1px solid gray; background-color: #f9f9f9;")
        image = QImage.fromData(buf.getvalue()); pixmap = QPixmap.fromImage(image)
        chart_area.setPixmap(pixmap); chart_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chart_area.setScaledContents(True)

        # Create percentage change label
        percent_label = QLabel(f"{'+' if last_price < now_price else '-'}{price_change}%")
        percent_label.setStyleSheet(f"font-size: 18px; border: none; color: {colour}")
        percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter); percent_label.setFixedWidth(120)

        # Create ticker label and add all parts to main layout
        ticker_label = QLabel(ticker)
        ticker_label.setStyleSheet("font-size: 24px; font-weight: bold; border: none;")
        ticker_label.setFixedWidth(120)
        add_to_layout(layout, [ticker_label, chart_area, percent_label], size_ratios=[0,1,0])

        # Set main styling and function to call on click
        card.setCursor(Qt.CursorShape.PointingHandCursor)
        card.mousePressEvent = lambda event, t=ticker, i=index: self.show_stock_menu(t, i)
        return card

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

    # Helper function to retrieve the logged in profile's data and username
    def get_profile_data(self) -> dict: return self.logged_profile.get_full_data()

    # Displays a menu to move or delete a stock
    def show_stock_menu(self, ticker: str, current_index: int) -> None:
        # Menu styling
        menu = QMenu(self); menu.setStyleSheet("QMenu {border: 1px solid black; padding: 5px}")
        move_up = menu.addAction("Move Up")
        move_down = menu.addAction("Move Down")
        delete_stock = menu.addAction("Delete from Saved")

        # Disable Move Up if at top, Move Down if at bottom
        if current_index == 0: move_up.setEnabled(False)
        if current_index == len(self.logged_profile.get_data().get("Saved stocks", [])) - 1:
            move_down.setEnabled(False)

        # Call correct helper function
        action = menu.exec(QCursor().pos())
        if action == move_up: self.reorder_stock(current_index, current_index - 1)
        elif action == move_down: self.reorder_stock(current_index, current_index + 1)
        elif action == delete_stock: self.remove_stock(ticker)

    # Helper function to swap stock positions in the list
    def reorder_stock(self, old_idx: int, new_idx: int) -> None:
        stocks = self.logged_profile.get_data().get("Saved stocks", [])
        stocks[old_idx], stocks[new_idx] = stocks[new_idx], stocks[old_idx]
        self.logged_profile.update_data({"Saved stocks": stocks})
        self.rebuild_frame("bottom")

    # Helper function to remove a stock from the list
    def remove_stock(self, ticker: str) -> None:
        current_data = self.logged_profile.get_data()
        current_data["Saved stocks"] = [s for s in current_data.get("Saved stocks", []) if s != ticker]
        self.logged_profile.update_data(current_data)
        self.rebuild_frame("bottom")

    # Helper function to add a stock to list
    def add_stock(self) -> None:
        # Validation on input
        if not self.search_input.text(): return
        ticker = self.search_input.text().upper()
        if not validate_ticker(ticker): QMessageBox.critical(self, "Failed", "Invalid ticker."); return

        # Add to the front of the list
        current_data = self.logged_profile.get_data().get("Saved stocks", [])
        current_data.insert(0, ticker)
        self.logged_profile.update_data({"Saved stocks": current_data})
        self.rebuild_frame("bottom")

    # Helper function to log out of profile and return to main menu
    def logout(self) -> None:
        self.parent_window.logged_profile = None; self.parent_window.logged_in = False
        self.parent_window.status_label.setText(f"Not logged in")
        self.rebuild_parent = True
        QMessageBox.information(self, "Success", "Logged out.")
        self.accept()

    # Helper function to change logged in profile
    def change_profile(self) -> None:
        # Get username and password input
        username, ok = QInputDialog.getText(self, "Login", "Enter Username:")
        if not ok: return

        password, ok = QInputDialog.getText(self, "Login", f"Enter Password:", QLineEdit.EchoMode.Password)
        if not ok: return

        result = self.parent_window.data_manager.get_profile(username, password)
        # Change logged in profile upon success, and rebuild right frame to update with user information
        if isinstance(result, Profile):
            self.parent_window.logged_profile = result
            self.parent_window.status_label.setText(f"Status: Logged In as {username}")
            self.rebuild_parent = True
            QMessageBox.information(self, "Success", f"Welcome, {username}!")
            self.accept()

        # Display error to user upon failure
        elif result == "Non-existent profile":
            QMessageBox.critical(
                self, "Profile Not Found", "Profile does not exist. Go back to main menu to create new."
            )

        elif result == "Incorrect password": QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        else: QMessageBox.critical(self, "Error", f"An error occurred: {result}")

    # Helper function to delete a profile
    def delete_profile(self) -> None:
        # Get password input
        password, ok = QInputDialog.getText(
            self, "Security check", f"Enter Password for {self.logged_profile.get_username()}:",
            QLineEdit.EchoMode.Password
        )
        if not ok: return

        # Give second confirmation to user
        confirmation = QMessageBox.question(
            self, "Confirm Action", "Are you sure you want to proceed? All data will be permanently lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )
        if confirmation != QMessageBox.StandardButton.Yes: return

        success = self.parent_window.data_manager.delete_profile(self.logged_profile, password)
        # Delete user data and logout of windows upon success
        if success is True:
            self.parent_window.logged_in = False; self.parent_window.logged_profile = None
            self.rebuild_parent = True
            self.parent_window.status_label.setText("Not logged in")
            QMessageBox.information(self, "Success", "Profile deleted.")
            self.accept()

        # Display error to user upon failure
        elif success == "Incorrect password": QMessageBox.critical(self, "Login Failed", "Incorrect password.")
        else: QMessageBox.critical(self, "Error", f"Failed to delete data.")

    # Helper function to export user data
    def export_profile(self) -> None:
        # Open the "Save As" Dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile Data",
            "profile_export.json",  # Default filename
            "JSON Files (*.json);;All Files (*)" # File filters
        )

        # Ensure user selected a path (didn't click "Cancel")
        if not file_path: return
        try:
            # Export data to chosen path
            data_to_export = self.logged_profile.get_data()
            with open(file_path, "w") as f: json.dump(data_to_export, f, indent=4)
            QMessageBox.information(self, "Success", f"Data exported to {file_path}")

        # Display error on failure
        except PermissionError: QMessageBox.critical(self, "Export Error", "Missing permissions.")
        except OSError: QMessageBox.critical(
            self, "Export Error", f"Failed to export data. Disk may be full or file is locked by another process."
        )
        except Exception as e: QMessageBox.critical(self, "Export Error", f"Could not save file: {e}")

    # Helper function to import data from a json
    def import_profile(self) -> None:
        # Open the "Open File" Dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile Data",
            "",  # Default directory
            "JSON Files (*.json);;All Files (*)" # File filters
        )

        # Ensure user selected a path (didn't click "Cancel")
        if not file_path: return
        try:
            # Import data and update profile
            with open(file_path, "r") as f: imported_data = json.load(f)
            self.logged_profile.update_data(imported_data)
            QMessageBox.information(self, "Success", "Profile data imported and saved successfully.")
            self.accept()

        # Display error on failure
        except json.JSONDecodeError: QMessageBox.critical(
            self, "Import Error", "The file is not a valid JSON file."
        )
        except Exception as e: QMessageBox.critical(self, "Import Error", f"Could not read file: {e}")

    # Helper function on profile label click to call correct function
    def label_click(self): self.choose_profile_icon()
    # Helper function to select a profile icon
    def choose_profile_icon(self):
        # Open the "Open File" Dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Choose pofile icon.",
            "",  # Default directory
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)" # File filters
        )

        # Ensure user selected a path (didn't click "Cancel")
        if not file_path: return
        try:
            # Get path of image and ensure another icon file doesn't already exist for that user
            _, ext = os.path.splitext(file_path)
            dest_path = os.path.join(
                ICON_DIR, f"{self.logged_profile.get_username()}{ext.lower()}"
            ).replace("\\", "/")

            if os.path.exists(dest_path): os.remove(dest_path)

            # Save a scaled version of the image in the correct path and return
            img = QPixmap(file_path).scaled(70,70); img.save(dest_path)
            self.rebuild_frame("top"); self.rebuild_parent = True
            QMessageBox.information(self, "Success", "Profile icon imported and saved successfully.")

        # Display error on failure
        except Exception as e: QMessageBox.critical(self, "Import Error", f"Could not read file: {e}")

    # Show parent when window is closed and update profile if needed
    def show_parent_on_close(self):
        if self.risk_slider.value() != self.logged_profile.get_data().get("Risk tolerance"):
            self.logged_profile.update_data({"Risk tolerance": self.risk_slider.value()})
        if self.rebuild_parent: self.parent_window.rebuild_frame("right")
        self.parent_window.show()

############################################################################

