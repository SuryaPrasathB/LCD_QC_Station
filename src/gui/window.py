from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from camera.worker import CameraWorker

class MainWindow(QMainWindow):
    def __init__(self, camera_impl):
        super().__init__()
        self.camera_impl = camera_impl
        self.worker = None

        self.init_ui()
        self.start_camera()

    def init_ui(self):
        self.setWindowTitle("Camera Live View")
        self.resize(800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label for Camera Feed
        self.image_label = QLabel("Initializing Camera...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: black; color: white; border: 1px solid #444; }")

        # Add to layout
        layout.addWidget(self.image_label)

    def start_camera(self):
        self.worker = CameraWorker(self.camera_impl)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    @pyqtSlot(QImage)
    def update_image(self, q_img):
        """Updates the label with the new frame, scaling it to fit."""
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)

        # Get current label size
        w = self.image_label.width()
        h = self.image_label.height()

        # Scale
        scaled_pixmap = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    @pyqtSlot(str)
    def handle_error(self, error_msg):
        # Only show one error at a time or log it
        print(f"Camera Error: {error_msg}")

    def closeEvent(self, event):
        """Handle clean shutdown."""
        if self.worker:
            self.image_label.setText("Stopping Camera...")
            self.worker.stop()
        event.accept()
