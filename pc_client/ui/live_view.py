from typing import Optional
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QPaintEvent

class LiveView(QWidget):
    # Signal emitted when a new ROI is drawn: x, y, w, h (normalized 0.0-1.0)
    roi_drawn = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True) # Only if we want hover effects, but mostly for drag

        self.pixmap: Optional[QPixmap] = None
        self.drawing = False
        self.setup_mode = False

        self.start_point: QPoint = QPoint()
        self.current_point: QPoint = QPoint()
        self.image_rect: QRectF = QRectF() # The rect where the image is actually drawn (letterboxed)

    def set_frame(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()

    def set_setup_mode(self, enabled: bool):
        self.setup_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000")) # Black background

        if self.pixmap and not self.pixmap.isNull():
            # Calculate aspect ratio scaling
            w_widget = self.width()
            h_widget = self.height()

            # Scale preserving aspect ratio
            scaled_pixmap = self.pixmap.scaled(
                w_widget, h_widget,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # Center the image
            x_offset = (w_widget - scaled_pixmap.width()) / 2
            y_offset = (h_widget - scaled_pixmap.height()) / 2

            painter.drawPixmap(int(x_offset), int(y_offset), scaled_pixmap)

            # Store image drawing rect for coordinate mapping
            self.image_rect = QRectF(x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())

            # Draw temporary ROI if dragging
            if self.drawing and self.setup_mode:
                pen = QPen(QColor(255, 255, 0), 2) # Yellow
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)

                rect = QRectF(self.start_point, self.current_point).normalized()
                painter.drawRect(rect)
        else:
            # Placeholder text
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")

    def mousePressEvent(self, event):
        if not self.setup_mode or not self.pixmap:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # Check if inside image rect
            if self.image_rect.contains(event.position()):
                self.drawing = True
                self.start_point = event.position().toPoint()
                self.current_point = self.start_point
                self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and self.setup_mode:
            # Clamp to image rect
            pos = event.position()
            x = max(self.image_rect.left(), min(pos.x(), self.image_rect.right()))
            y = max(self.image_rect.top(), min(pos.y(), self.image_rect.bottom()))
            self.current_point = QPoint(int(x), int(y))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.setup_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.update()

            # Normalize coordinates
            rect = QRectF(self.start_point, self.current_point).normalized()

            # Check for minimum size (ignore accidental clicks)
            if rect.width() < 5 or rect.height() < 5:
                return

            # Convert to 0.0 - 1.0 relative to the image
            x_norm = (rect.x() - self.image_rect.x()) / self.image_rect.width()
            y_norm = (rect.y() - self.image_rect.y()) / self.image_rect.height()
            w_norm = rect.width() / self.image_rect.width()
            h_norm = rect.height() / self.image_rect.height()

            # Clamp to 0-1 (just in case)
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            self.roi_drawn.emit(x_norm, y_norm, w_norm, h_norm)
