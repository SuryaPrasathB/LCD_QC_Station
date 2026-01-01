from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal

class ResultsPanel(QWidget):
    # Signals for overrides (now specific to ROI)
    # Action, ROI_ID (None for global, but we only do ROI now)
    override_action = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_status = None
        self.lbl_meta = None
        self.table = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Status Group
        status_group = QGroupBox("Inspection Status")
        status_layout = QVBoxLayout()

        self.lbl_status = QLabel("WAITING")
        self.lbl_status.setObjectName("result_waiting")
        status_layout.addWidget(self.lbl_status)

        self.lbl_meta = QLabel("Dataset: - | Model: -")
        self.lbl_meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.lbl_meta)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # ROI Results Table
        table_group = QGroupBox("ROI Results")
        table_layout = QVBoxLayout()

        self.table = QTableWidget()
        # Columns: ID, Status, Score, Reason, Override (Pass/Fail)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ROI ID", "Status", "Score", "Reason", "Override"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 150)

        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Note: Global override buttons removed as per request.

    def update_results(self, result: dict):
        """Updates the panel with inspection results."""
        if not result:
            self.lbl_status.setText("WAITING")
            self.lbl_status.setObjectName("result_waiting")
            self.lbl_meta.setText("Dataset: - | Model: -")
            self.table.setRowCount(0)
            self.style().unpolish(self.lbl_status)
            self.style().polish(self.lbl_status)
            return

        # Determine effective pass status
        original_passed = result.get("passed", False)
        is_overridden = result.get("overridden", False)

        passed = original_passed
        status_text = ""

        if is_overridden:
            override_status = result.get("override_status", "").upper()
            if override_status == "PASS":
                passed = True
                status_text = "PASS (OVR)"
            elif override_status == "FAIL":
                passed = False
                status_text = "FAIL (OVR)"
            else:
                # Fallback if unknown status
                status_text = "PASS (OVR)" if original_passed else "FAIL (OVR)"
        else:
            status_text = "PASS" if original_passed else "FAIL"

        self.lbl_status.setText(status_text)
        self.lbl_status.setObjectName("result_pass" if passed else "result_fail")

        # Force style refresh
        self.style().unpolish(self.lbl_status)
        self.style().polish(self.lbl_status)

        # Update Metadata
        dataset = result.get("dataset_version", "N/A")
        model = result.get("model_version", "N/A")
        self.lbl_meta.setText(f"Dataset: {dataset} | Model: {model}")

        # Update Table
        roi_results = result.get("roi_results", {})
        self.table.setRowCount(len(roi_results))

        for row, (roi_id, res) in enumerate(roi_results.items()):
            roi_pass = res.get("passed", False)
            score = res.get("score", 0.0)
            reason = res.get("failure_reason")
            detail = res.get("failure_detail")

            reason_text = ""
            if not roi_pass:
                if reason:
                    reason_text = reason
                    if detail:
                        reason_text += f" ({detail})"
                else:
                    reason_text = "Similarity Fail"

            item_id = QTableWidgetItem(roi_id)

            item_status = QTableWidgetItem("PASS" if roi_pass else "FAIL")
            item_status.setForeground(Qt.GlobalColor.green if roi_pass else Qt.GlobalColor.red)
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_score = QTableWidgetItem(f"{score:.4f}")
            item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_reason = QTableWidgetItem(reason_text)
            item_reason.setToolTip(reason_text)

            self.table.setItem(row, 0, item_id)
            self.table.setItem(row, 1, item_status)
            self.table.setItem(row, 2, item_score)
            self.table.setItem(row, 3, item_reason)

            # Override Buttons (Sub-column)
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(2, 2, 2, 2)

            btn_p = QPushButton("PASS")
            btn_p.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; font-size: 10px; padding: 2px;")
            btn_p.clicked.connect(lambda checked, a="pass", r=roi_id: self.override_action.emit(a, r))

            btn_f = QPushButton("FAIL")
            btn_f.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 10px; padding: 2px;")
            btn_f.clicked.connect(lambda checked, a="fail", r=roi_id: self.override_action.emit(a, r))

            btn_layout.addWidget(btn_p)
            btn_layout.addWidget(btn_f)

            self.table.setCellWidget(row, 4, btn_widget)

    def set_buttons_enabled(self, enabled: bool):
        # We need to iterate over rows and disable/enable cell widgets
        for row in range(self.table.rowCount()):
             widget = self.table.cellWidget(row, 4)
             if widget:
                 widget.setEnabled(enabled)

    def clear(self):
        self.update_results({})
