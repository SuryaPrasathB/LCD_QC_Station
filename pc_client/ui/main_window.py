from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QInputDialog
)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, QEvent
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
            print(f"[Client] Worker Error in {self.func.__name__}: {e}")
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
        self.current_inspection_id = None # Store ID for overrides

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
        print("[Client] Application initialized")

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
        self.results_panel.override_pass.connect(lambda: self.trigger_override("pass"))
        self.results_panel.override_fail.connect(lambda: self.trigger_override("fail"))
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

    def closeEvent(self, event):
        print("[Client] Closing application...")
        # Stop timers
        self.live_timer.stop()
        self.result_timer.stop()

        # Stop polling workers if active
        if self.frame_worker and self.frame_worker.isRunning():
            self.frame_worker.wait(500)
            self.frame_worker.terminate()

        if self.result_worker and self.result_worker.isRunning():
            self.result_worker.wait(500)
            self.result_worker.terminate()

        # Wait for all task workers
        # We can iterate over copy because threads remove themselves from set
        for worker in list(self.active_workers):
            if worker.isRunning():
                worker.wait(500)
                # Force kill if stuck (e.g. network timeout)
                if worker.isRunning():
                    worker.terminate()

        super().closeEvent(event)

    def apply_state(self, connected: bool):
        self.ctrl_group.setEnabled(connected)
        self.txt_ip.setEnabled(not connected)
        self.txt_port.setEnabled(not connected)
        self.results_panel.set_buttons_enabled(connected and self.current_inspection_id is not None)

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
            self.frame_worker.start()
        except Exception as e: # pylint: disable=broad-exception-caught
            # print(f"Poll Error: {e}") # Optional
            pass

    def update_live_view(self, data):
        pixmap = bytes_to_pixmap(data)
        self.live_view.set_frame(pixmap)

    def toggle_setup_mode(self, checked):
        print(f"[Client] Setup Mode: {checked}")
        self.setup_mode_active = checked
        self.live_view.set_setup_mode(checked)
        if checked:
            self.btn_setup.setText("Exit Setup")
        else:
            self.btn_setup.setText("Setup ROI")

    def on_roi_drawn(self, x, y, w, h):
        print(f"[Client] ROI drawn: {x:.3f},{y:.3f},{w:.3f},{h:.3f}")
        worker = self.start_worker(self.client.get_roi_list)
        worker.result_ready.connect(lambda rois: self.prompt_roi_name(rois, x, y, w, h))

    def prompt_roi_name(self, current_rois, x, y, w, h):
        default_id = f"roi_{len(current_rois) + 1}"
        name, ok = QInputDialog.getText(self, "ROI ID", "Enter ROI ID:", text=default_id)
        if ok and name:
            self.add_roi(name, x, y, w, h)

    def add_roi(self, roi_id, x, y, w, h):
        worker = self.start_worker(self.client.set_roi, roi_id, x, y, w, h)
        worker.result_ready.connect(lambda res: print(f"[Client] ROI {roi_id} added"))

    def clear_rois(self):
        print("[Client] Clearing ROIs requested")
        self.start_worker(self.client.clear_rois)

    def commit_rois(self):
        print("[Client] Committing ROIs requested")
        worker = self.start_worker(self.client.commit_rois)
        worker.result_ready.connect(lambda: QMessageBox.information(self, "Info", "ROIs Committed"))

    def start_inspection(self):
        print("[Client] Start Inspection requested")
        self.results_panel.clear()
        self.current_inspection_id = None
        self.results_panel.set_buttons_enabled(False)

        worker = self.start_worker(self.client.start_inspection)
        worker.result_ready.connect(self.on_inspection_started)
        worker.error_occurred.connect(lambda e: QMessageBox.warning(self, "Error", f"Busy: {e}"))

    def on_inspection_started(self, insp_id):
        self.inspection_pending = True
        self.current_inspection_id = insp_id
        self.result_timer.start()

    def poll_inspection_result(self):
        if self.result_worker and self.result_worker.isRunning():
            return

        self.result_worker = ApiWorker(self.client.get_inspection_result)
        self.result_worker.result_ready.connect(self.on_inspection_result)
        self.result_worker.error_occurred.connect(self.on_poll_error)
        self.result_worker.start()

    def on_poll_error(self, err_msg):
        # print(f"Poll Result Error: {err_msg}")
        pass

    def on_inspection_result(self, result):
        res_id = result.get('inspection_id')
        print(f"[Client] Polled Result ID: {res_id} (Expected: {self.current_inspection_id})")

        if self.current_inspection_id and res_id != self.current_inspection_id:
            print("[Client] Received stale result. Ignoring and continuing poll...")
            return

        self.results_panel.update_results(result)
        self.results_panel.set_buttons_enabled(True) # Enable override buttons
        self.result_timer.stop()
        self.inspection_pending = False
        self.fetch_inspection_frame()

    def fetch_inspection_frame(self):
        worker = self.start_worker(self.client.get_inspection_frame)
        worker.result_ready.connect(self.update_live_view)

    def trigger_override(self, action):
        if not self.current_inspection_id:
            return

        print(f"[Client] Triggering Override: {action}")
        worker = self.start_worker(self.client.override_inspection, self.current_inspection_id, action)
        worker.result_ready.connect(lambda: QMessageBox.information(self, "Override", f"Marked as {action.upper()}"))
