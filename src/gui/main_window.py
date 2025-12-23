import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter,
                             QTextEdit, QStatusBar)
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from src.core.config import settings
from src.core.engine import InspectionEngine
from src.hal.files.source import FileImageSource
from src.core.types import InspectionResultType
from src.infra.logging import get_logger

logger = get_logger(__name__)

class InspectionWorker(QThread):
    result_ready = pyqtSignal(object, object) # result, raw_image

    def __init__(self, engine, source):
        super().__init__()
        self.engine = engine
        self.source = source

    def run(self):
        try:
            raw = self.source.capture()
            result = self.engine.process(raw)
            self.result_ready.emit(result, raw)
        except Exception as e:
            logger.error(f"GUI Inspection Error: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LCD Inspection System - Developer Mode")
        self.resize(1200, 800)

        self.engine = InspectionEngine()
        # Default to a safe path initially
        self.source = FileImageSource(settings.IMAGE_STORE_PATH)
        self.source.initialize()

        self._init_ui()

    def _init_ui(self):
        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main Layout
        layout = QVBoxLayout(central)

        # Toolbar
        toolbar = QHBoxLayout()

        btn_load = QPushButton("Load Image Folder")
        btn_load.clicked.connect(self.load_folder)
        toolbar.addWidget(btn_load)

        btn_inspect = QPushButton("Run Inspection (Next Image)")
        btn_inspect.clicked.connect(self.run_inspection)
        toolbar.addWidget(btn_inspect)

        self.lbl_status = QLabel("Ready")
        toolbar.addWidget(self.lbl_status)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Splitter for Image vs Info
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image Viewer
        self.lbl_image = QLabel("No Image")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setStyleSheet("background-color: #333; color: #fff;")
        self.lbl_image.setMinimumSize(640, 480)
        splitter.addWidget(self.lbl_image)

        # Info Panel
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        info_layout.addWidget(QLabel("Inspection Log:"))
        info_layout.addWidget(self.txt_log)

        # Actions
        self.btn_override = QPushButton("Mark as FALSE FAIL")
        self.btn_override.setStyleSheet("background-color: #d9534f; color: white;")
        self.btn_override.setEnabled(False) # Disabled until we have a result
        self.btn_override.clicked.connect(self.mark_false_fail)
        info_layout.addWidget(self.btn_override)

        splitter.addWidget(info_panel)
        layout.addWidget(splitter)

        # Status Bar
        self.setStatusBar(QStatusBar())

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.source = FileImageSource(Path(folder))
            self.source.initialize()
            self.lbl_status.setText(f"Loaded: {folder}")
            self.txt_log.append(f"Source changed to: {folder}")

    def run_inspection(self):
        self.lbl_status.setText("Inspecting...")
        self.worker = InspectionWorker(self.engine, self.source)
        self.worker.result_ready.connect(self.on_inspection_complete)
        self.worker.start()

    def on_inspection_complete(self, result, raw_image):
        self.last_result = result # Store for override
        self.btn_override.setEnabled(True)

        # Display Image
        h, w, ch = raw_image.data.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB
        rgb_data = cv2.cvtColor(raw_image.data, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.lbl_image.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio
        ))

        # Log Result
        msg = f"ID: {result.image_id}\nResult: {result.result}\nScore: {result.confidence_score:.3f}\nTime: {result.processing_time_ms}ms"
        self.txt_log.append("----------------")
        self.txt_log.append(msg)

        status_color = "green" if result.result == InspectionResultType.PASS else "red"
        self.lbl_status.setText(f"Result: {result.result}")
        self.lbl_status.setStyleSheet(f"color: {status_color}; font-weight: bold;")

    def mark_false_fail(self):
        if not hasattr(self, 'last_result'):
            return

        # In a real app we might use a shared DatasetManager instance or API client.
        # Since GUI mode imports core directly:
        from src.core.dataset import DatasetManager
        dm = DatasetManager()
        dm.add_override(self.last_result.image_id, "FAIL", "PASS", reason="User marked as False Fail in GUI")
        self.txt_log.append(">> Override marked (Pending)")
        self.btn_override.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
