from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox
)
from PyQt6.QtCore import Qt

class ResultsPanel(QWidget):
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
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["ROI ID", "Status", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

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

        # Update Global Status
        passed = result.get("passed", False)
        self.lbl_status.setText("PASS" if passed else "FAIL")
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

            item_id = QTableWidgetItem(roi_id)

            item_status = QTableWidgetItem("PASS" if roi_pass else "FAIL")
            item_status.setForeground(Qt.GlobalColor.green if roi_pass else Qt.GlobalColor.red)

            item_score = QTableWidgetItem(f"{score:.4f}")

            self.table.setItem(row, 0, item_id)
            self.table.setItem(row, 1, item_status)
            self.table.setItem(row, 2, item_score)

    def clear(self):
        self.update_results({})
