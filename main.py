"""Image Tool: ImageJ-like batch image analysis tool."""

import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
