from typing import Optional
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
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

        self.start_point: QPointF = QPointF()
        self.current_point: QPointF = QPointF()
        self.image_rect: QRectF = QRectF() # The rect where the image is actually drawn (letterboxed)
        
        # Zoom and Pan
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.panning = False
        self.last_pan_point = QPointF()

    def set_frame(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()

    def set_setup_mode(self, enabled: bool):
        self.setup_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def wheelEvent(self, event):
        if not self.pixmap or self.pixmap.isNull():
            return
            
        zoom_step = 0.1
        if event.angleDelta().y() > 0:
            self.zoom_factor += zoom_step
        else:
            self.zoom_factor -= zoom_step
            
        self.zoom_factor = max(1.0, min(self.zoom_factor, 5.0))
        
        # If zooming out all the way, reset pan
        if self.zoom_factor == 1.0:
            self.pan_offset = QPointF(0, 0)
            
        self.update()

    def map_to_screen(self, p: QPointF) -> QPointF:
        center = QPointF(self.width() / 2, self.height() / 2)
        x = (p.x() - center.x()) * self.zoom_factor + center.x() + self.pan_offset.x() * self.zoom_factor
        y = (p.y() - center.y()) * self.zoom_factor + center.y() + self.pan_offset.y() * self.zoom_factor
        return QPointF(x, y)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000")) # Black background

        if self.pixmap and not self.pixmap.isNull():
            painter.save()
            
            # Translate to center to scale, then translate back
            center = QPointF(self.width() / 2, self.height() / 2)
            painter.translate(center)
            painter.scale(self.zoom_factor, self.zoom_factor)
            painter.translate(-center)
            
            # Apply pan
            painter.translate(self.pan_offset)

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
            
            painter.restore()

            # Draw temporary ROI if dragging
            if self.drawing and self.setup_mode:
                pen = QPen(QColor(255, 255, 0), 2) # Yellow
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)

                # Robust casting to QPointF
                p1 = QPointF(self.start_point)
                p2 = QPointF(self.current_point)
                
                p1_screen = self.map_to_screen(p1)
                p2_screen = self.map_to_screen(p2)
                
                rect = QRectF(p1_screen, p2_screen).normalized()
                painter.drawRect(rect)
        else:
            # Placeholder text
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Signal")

    def map_to_image(self, pos: QPointF) -> QPointF:
        """Map widget coordinates to scaled/panned image coordinates."""
        # Inverse of paintEvent transforms
        center = QPointF(self.width() / 2, self.height() / 2)
        
        # 1. Remove pan
        x = pos.x() - self.pan_offset.x() * self.zoom_factor
        y = pos.y() - self.pan_offset.y() * self.zoom_factor
        
        # 2. Remove zoom around center
        x = (x - center.x()) / self.zoom_factor + center.x()
        y = (y - center.y()) / self.zoom_factor + center.y()
        
        return QPointF(x, y)

    def mousePressEvent(self, event):
        if not self.pixmap:
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.panning = True
            self.last_pan_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if not self.setup_mode:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            mapped_pos = self.map_to_image(event.position())
            # Check if inside image rect
            if self.image_rect.contains(mapped_pos):
                self.drawing = True
                self.start_point = mapped_pos
                self.current_point = self.start_point
                self.update()

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.position() - self.last_pan_point
            # Scale pan distance inversely by zoom factor so it feels 1:1 with mouse
            self.pan_offset += QPointF(delta.x() / self.zoom_factor, delta.y() / self.zoom_factor)
            self.last_pan_point = event.position()
            self.update()
            return

        if self.drawing and self.setup_mode:
            mapped_pos = self.map_to_image(event.position())
            # Clamp to image rect
            x = max(self.image_rect.left(), min(mapped_pos.x(), self.image_rect.right()))
            y = max(self.image_rect.top(), min(mapped_pos.y(), self.image_rect.bottom()))
            self.current_point = QPointF(x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self.setup_mode else Qt.CursorShape.ArrowCursor)
            return

        if self.drawing and self.setup_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.update()

            # Robust casting to QPointF
            p1 = QPointF(self.start_point)
            p2 = QPointF(self.current_point)
            rect = QRectF(p1, p2).normalized()

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
