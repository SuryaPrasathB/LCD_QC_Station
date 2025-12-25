from enum import Enum, auto
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from camera.worker import CameraWorker, CaptureWorker
from .video_label import VideoLabel

class AppState(Enum):
    LIVE_VIEW = auto()
    CAPTURED = auto()

class MainWindow(QMainWindow):
    def __init__(self, camera_impl):
        super().__init__()
        self.camera_impl = camera_impl
        self.worker = None
        self.capture_worker = None
        self.current_state = None

        self.init_ui()

        # Start in LIVE_VIEW
        self.set_state(AppState.LIVE_VIEW)

    def init_ui(self):
        self.setWindowTitle("Camera Live View")
        self.resize(800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label for Camera Feed
        self.image_label = VideoLabel("Initializing Camera...")
        layout.addWidget(self.image_label)

        # Buttons Layout
        btn_layout = QHBoxLayout()

        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(self.handle_capture_click)
        btn_layout.addWidget(self.btn_capture)

        self.btn_resume = QPushButton("Resume Live")
        self.btn_resume.clicked.connect(self.handle_resume_click)
        btn_layout.addWidget(self.btn_resume)

        layout.addLayout(btn_layout)

    def set_state(self, state: AppState):
        self.current_state = state

        if state == AppState.LIVE_VIEW:
            self.btn_capture.setEnabled(True)
            self.btn_resume.setEnabled(False)
            self.start_live_view()

        elif state == AppState.CAPTURED:
            self.btn_capture.setEnabled(False)
            self.btn_resume.setEnabled(True)
            # Live view is stopped in the transition logic before reaching here

    def start_live_view(self):
        # Stop existing worker if any (sanity check)
        if self.worker:
            self.worker.stop()

        self.image_label.setText("Starting Live View...")
        self.worker = CameraWorker(self.camera_impl)
        self.worker.frame_ready.connect(self.update_image)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_capture_click(self):
        """
        Transition from LIVE_VIEW to CAPTURED (via async capture).
        """
        if self.current_state != AppState.LIVE_VIEW:
            return

        # 1. Disable buttons to prevent re-entry
        self.btn_capture.setEnabled(False)
        self.btn_resume.setEnabled(False)

        # 2. Stop Live View
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.image_label.setText("Capturing...")

        # 3. Start Capture Worker
        self.capture_worker = CaptureWorker(self.camera_impl, "/tmp/capture.jpg")
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.error_occurred.connect(self.on_capture_error)
        self.capture_worker.start()

    def on_capture_finished(self):
        # 4. Load and display image
        pixmap = QPixmap("/tmp/capture.jpg")
        if not pixmap.isNull():
            self.image_label.set_frame(pixmap)
            # 5. Update State
            self.set_state(AppState.CAPTURED)
        else:
            self.on_capture_error("Failed to load captured image.")

    def on_capture_error(self, error_msg):
        # Show error
        QMessageBox.critical(self, "Capture Error", f"Capture failed:\n{error_msg}")
        # Revert to Live View
        self.set_state(AppState.LIVE_VIEW)

    def handle_resume_click(self):
        if self.current_state != AppState.CAPTURED:
            return

        self.image_label.setText("Resuming...")
        self.set_state(AppState.LIVE_VIEW)

    @pyqtSlot(QImage)
    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.set_frame(pixmap)

    @pyqtSlot(str)
    def handle_error(self, error_msg):
        print(f"Camera Error: {error_msg}")
        # Could show a message box if critical, but for live stream usually just log

    def closeEvent(self, event):
        if self.worker:
            self.image_label.setText("Stopping Camera...")
            self.worker.stop()
        if self.capture_worker and self.capture_worker.isRunning():
            self.capture_worker.quit()
            self.capture_worker.wait()
        event.accept()
