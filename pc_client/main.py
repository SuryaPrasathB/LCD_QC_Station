import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QFile, QTextStream

# Ensure we can import from pc_client
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pc_client.ui.main_window import MainWindow # pylint: disable=wrong-import-position

def main():
    app = QApplication(sys.argv)

    # Load stylesheet
    style_file = QFile(os.path.join(os.path.dirname(__file__), "styles/theme.qss"))
    if style_file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        stream = QTextStream(style_file)
        app.setStyleSheet(stream.readAll())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
