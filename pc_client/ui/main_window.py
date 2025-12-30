from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QInputDialog
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal
from pc_client.api.inspection_api import InspectionClient
from pc_client.ui.live_view import LiveView
from pc_client.ui.results_panel import ResultsPanel
from pc_client.utils.image_utils import bytes_to_pixmap

class ApiWorker(QThread):
    """Worker thread for non-blocking API calls."""
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    finished_task = pyqtSignal(object) # Emit self to be removed from pool

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.func(*self.args, **self.kwargs)
            self.result_ready.emit(res)
        except Exception as e: # pylint: disable=broad-exception-caught
            self.error_occurred.emit(str(e))
        finally:
            self.finished_task.emit(self)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LCD Inspection Client")
        self.resize(1024, 768)

        self.client = InspectionClient()
        self.setup_mode_active = False
        self.inspection_pending = False

        # Worker pool to prevent GC
        self.active_workers = set()

        # Specific workers for polling (reused/managed separately)
        self.frame_worker = None
        self.result_worker = None

        # Poll Timers
        self.live_timer = QTimer()
        self.live_timer.setInterval(100) # 10 FPS
        self.live_timer.timeout.connect(self.poll_live_feed)

        self.result_timer = QTimer()
        self.result_timer.setInterval(500) # 2 FPS
        self.result_timer.timeout.connect(self.poll_inspection_result)

        self.init_ui()
        self.apply_state(connected=False)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column (Live View)
        left_layout = QVBoxLayout()

        # Connection Bar
        conn_group = QGroupBox("Connection")
        conn_layout = QHBoxLayout()

        self.txt_ip = QLineEdit("127.0.0.1")
        self.txt_ip.setPlaceholderText("IP Address")
        self.txt_port = QLineEdit("8000")
        self.txt_port.setPlaceholderText("Port")
        self.txt_port.setFixedWidth(60)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setObjectName("action_button")
        self.btn_connect.clicked.connect(self.toggle_connection)

        self.lbl_conn_status = QLabel("● Disconnected")
        self.lbl_conn_status.setObjectName("status_disconnected")

        conn_layout.addWidget(QLabel("IP:"))
        conn_layout.addWidget(self.txt_ip)
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.txt_port)
        conn_layout.addWidget(self.btn_connect)
        conn_layout.addWidget(self.lbl_conn_status)
        conn_group.setLayout(conn_layout)

        left_layout.addWidget(conn_group)

        # Live View
        self.live_view = LiveView()
        self.live_view.roi_drawn.connect(self.on_roi_drawn)
        left_layout.addWidget(self.live_view)

        main_layout.addLayout(left_layout, stretch=2)

        # Right Column (Controls & Results)
        right_layout = QVBoxLayout()

        # Control Panel
        self.ctrl_group = QGroupBox("Control Panel")
        ctrl_layout = QVBoxLayout()

        self.btn_setup = QPushButton("Setup ROI")
        self.btn_setup.setCheckable(True)
        self.btn_setup.toggled.connect(self.toggle_setup_mode)

        self.btn_clear = QPushButton("Clear ROIs")
        self.btn_clear.setObjectName("danger_button")
        self.btn_clear.clicked.connect(self.clear_rois)

        self.btn_commit = QPushButton("Commit Changes")
        self.btn_commit.setObjectName("action_button")
        self.btn_commit.clicked.connect(self.commit_rois)

        self.btn_inspect = QPushButton("Start Inspection")
        self.btn_inspect.setStyleSheet("font-size: 16px; padding: 10px;")
        self.btn_inspect.clicked.connect(self.start_inspection)

        ctrl_layout.addWidget(self.btn_setup)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_commit)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(self.btn_inspect)
        self.ctrl_group.setLayout(ctrl_layout)

        right_layout.addWidget(self.ctrl_group)

        # Results Panel
        self.results_panel = ResultsPanel()
        right_layout.addWidget(self.results_panel)

        main_layout.addLayout(right_layout, stretch=1)

    def start_worker(self, func, *args, **kwargs):
        worker = ApiWorker(func, *args, **kwargs)
        self.active_workers.add(worker)
        worker.finished_task.connect(self.cleanup_worker)
        worker.start()
        return worker

    def cleanup_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)

    def apply_state(self, connected: bool):
        self.ctrl_group.setEnabled(connected)
        self.txt_ip.setEnabled(not connected)
        self.txt_port.setEnabled(not connected)

        if connected:
            self.btn_connect.setText("Disconnect")
            self.lbl_conn_status.setText("● Connected")
            self.lbl_conn_status.setObjectName("status_connected")
            self.live_timer.start()
        else:
            self.btn_connect.setText("Connect")
            self.lbl_conn_status.setText("● Disconnected")
            self.lbl_conn_status.setObjectName("status_disconnected")
            self.live_timer.stop()
            self.result_timer.stop()
            self.live_view.set_frame(None)
            self.results_panel.clear()

        # Refresh style
        self.style().unpolish(self.lbl_conn_status)
        self.style().polish(self.lbl_conn_status)

    def toggle_connection(self):
        if self.btn_connect.text() == "Disconnect":
            self.apply_state(False)
            return

        ip = self.txt_ip.text()
        try:
            port = int(self.txt_port.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid Port")
            return

        # Connect Worker
        worker = self.start_worker(self.client.connect, ip, port)
        worker.result_ready.connect(self.on_connect_result)
        self.btn_connect.setEnabled(False)

    def on_connect_result(self, success):
        self.btn_connect.setEnabled(True)
        if success:
            self.apply_state(True)
        else:
            QMessageBox.critical(self, "Error", "Could not connect to server.")

    def poll_live_feed(self):
        try:
            if self.frame_worker and self.frame_worker.isRunning():
                return

            self.frame_worker = ApiWorker(self.client.get_live_frame, with_rois=True)
            self.frame_worker.result_ready.connect(self.update_live_view)
            # We don't add polling workers to active_workers set to avoid churn,
            # but we hold a reference in self.frame_worker
            self.frame_worker.start()
        except Exception: # pylint: disable=broad-exception-caught
            pass

    def update_live_view(self, data):
        pixmap = bytes_to_pixmap(data)
        self.live_view.set_frame(pixmap)

    def toggle_setup_mode(self, checked):
        self.setup_mode_active = checked
        self.live_view.set_setup_mode(checked)
        if checked:
            self.btn_setup.setText("Exit Setup")
        else:
            self.btn_setup.setText("Setup ROI")

    def on_roi_drawn(self, x, y, w, h):
        # Fetch current list size to suggest ID (in worker or main? Fetching is blocking)
        # Using a worker to fetch list first would be robust but complex callback chain.
        # User is waiting for dialog. I'll just use a generic default and let user type.
        # Or better: I can't block UI to fetch list.
        # I will assume "new_roi" or user enters it.
        # Requirement: "Auto-generate ROI ID (roi_1, roi_2, …)"
        # I'll just use a timestamp or random suffix if I don't know the count?
        # No, I should respect the sequence.
        # I'll fire a worker to get the list, THEN show dialog?
        # Yes.
        worker = self.start_worker(self.client.get_roi_list)
        worker.result_ready.connect(lambda rois: self.prompt_roi_name(rois, x, y, w, h))

    def prompt_roi_name(self, current_rois, x, y, w, h):
        default_id = f"roi_{len(current_rois) + 1}"
        name, ok = QInputDialog.getText(self, "ROI ID", "Enter ROI ID:", text=default_id)
        if ok and name:
            self.add_roi(name, x, y, w, h)

    def add_roi(self, roi_id, x, y, w, h):
        worker = self.start_worker(self.client.set_roi, roi_id, x, y, w, h)
        worker.result_ready.connect(lambda res: print(f"ROI Added: {res}"))

    def clear_rois(self):
        self.start_worker(self.client.clear_rois)

    def commit_rois(self):
        worker = self.start_worker(self.client.commit_rois)
        worker.result_ready.connect(lambda: QMessageBox.information(self, "Info", "ROIs Committed"))

    def start_inspection(self):
        self.results_panel.clear()
        worker = self.start_worker(self.client.start_inspection)
        worker.result_ready.connect(self.on_inspection_started)
        worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Error", f"Busy: {e}"))

    def on_inspection_started(self, insp_id): # pylint: disable=unused-argument
        self.inspection_pending = True
        self.result_timer.start()

    def poll_inspection_result(self):
        if self.result_worker and self.result_worker.isRunning():
            return

        self.result_worker = ApiWorker(self.client.get_inspection_result)
        self.result_worker.result_ready.connect(self.on_inspection_result)
        # We handle error (e.g. 404 not found yet)
        self.result_worker.error_occurred.connect(self.on_poll_error)
        self.result_worker.start()

    def on_poll_error(self, err_msg):
        # If 404, maybe not ready? But API raises 404 if "No inspection found".
        # If we just started one, it should eventually exist?
        # Or maybe start_inspection failed silently?
        # We just keep polling for a bit or until user cancels?
        # For now, we just ignore errors and keep polling (or stop after max retries).
        pass

    def on_inspection_result(self, result):
        # We got a result.
        self.results_panel.update_results(result)
        self.result_timer.stop()
        self.inspection_pending = False
        self.fetch_inspection_frame()

    def fetch_inspection_frame(self):
        worker = self.start_worker(self.client.get_inspection_frame)
        worker.result_ready.connect(self.update_live_view)
