from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap
from pc_client.api.inspection_api import InspectionClient
from pc_client.utils.image_utils import bytes_to_pixmap

class ReferenceGalleryDialog(QDialog):
    def __init__(self, client: InspectionClient, roi_id: str, parent=None):
        super().__init__(parent)
        self.client = client
        self.roi_id = roi_id
        self.setWindowTitle(f"Reference Gallery - {roi_id}")
        self.resize(800, 600)
        self.init_ui()
        self.load_references()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel(f"References for ROI: {self.roi_id}"))

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.load_references)
        header_layout.addWidget(self.btn_refresh)

        self.btn_delete_all = QPushButton("Delete ALL")
        self.btn_delete_all.setObjectName("danger_button")
        self.btn_delete_all.setStyleSheet("background-color: #d32f2f; color: white;")
        self.btn_delete_all.clicked.connect(self.delete_all_references)
        header_layout.addWidget(self.btn_delete_all)

        layout.addLayout(header_layout)

        # Scroll Area for Grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll.setWidget(self.grid_widget)

        layout.addWidget(self.scroll)

        # Status
        self.lbl_status = QLabel("Ready")
        layout.addWidget(self.lbl_status)

    def load_references(self):
        self.lbl_status.setText("Loading...")
        # clear grid
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        try:
            # We need to implement list_references in Client API first!
            # Assuming I will update api/inspection_api.py next.
            # Using raw requests if not yet available, but better to update client first.
            # For now, I'll use the client method assuming it exists (I will add it).
            refs_data = self.client.list_roi_references(self.roi_id)
            refs = refs_data.get("references", [])

            if not refs:
                self.grid_layout.addWidget(QLabel("No references found."), 0, 0)
                self.lbl_status.setText("No references.")
                return

            row, col = 0, 0
            cols = 4

            for ref in refs:
                ref_id = ref["id"]
                filename = ref["filename"]

                # Container
                container = QWidget()
                vbox = QVBoxLayout(container)

                # Image
                lbl_img = QLabel("Loading...")
                lbl_img.setFixedSize(128, 128)
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_img.setStyleSheet("border: 1px solid gray;")
                vbox.addWidget(lbl_img)

                # Label
                vbox.addWidget(QLabel(filename))

                # Delete Btn
                btn_del = QPushButton("Delete")
                btn_del.clicked.connect(lambda checked, rid=ref_id: self.delete_reference(rid))
                vbox.addWidget(btn_del)

                self.grid_layout.addWidget(container, row, col)

                # Fetch Image in background?
                # For simplicity, fetch synchronously here or use a worker if I had one.
                # Since this is a dialog, simple fetch is okay for prototype.
                img_bytes = self.client.get_roi_reference_image(self.roi_id, ref_id)
                if img_bytes:
                    pix = bytes_to_pixmap(img_bytes)
                    if pix:
                        lbl_img.setPixmap(pix.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio))
                        lbl_img.setText("")

                col += 1
                if col >= cols:
                    col = 0
                    row += 1

            self.lbl_status.setText(f"Loaded {len(refs)} references.")

        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")
            print(f"Gallery Error: {e}")

    def delete_reference(self, ref_id):
        confirm = QMessageBox.question(self, "Confirm Delete", f"Delete reference {ref_id}?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.client.delete_roi_reference(self.roi_id, ref_id)
                self.load_references() # Reload
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))

    def delete_all_references(self):
        confirm = QMessageBox.question(self, "Confirm Delete ALL", f"Delete ALL references for {self.roi_id}? This cannot be undone.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                self.client.delete_all_roi_references(self.roi_id)
                self.load_references()
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
