from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen

class VideoLabel(QLabel):
    """
    A custom QLabel that draws a scaled pixmap in its paintEvent.
    Also handles ROI selection (mouse dragging) and display.
    """
    # Emits the current selected rect in IMAGE coordinates
    selection_changed = pyqtSignal(QRect)

    def __init__(self, text=""):
        super().__init__(text)
        self.pixmap_frame = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")

        # Selection State
        self.is_selection_enabled = False
        self.is_dragging = False
        self.start_point = QPoint()
        self.current_point = QPoint()

        # Current ROI in IMAGE coordinates
        self.roi_rect_img: QRect = QRect()

    def set_frame(self, pixmap: QPixmap):
        """Update the current frame and trigger a repaint."""
        self.pixmap_frame = pixmap
        self.update()

    def set_selection_enabled(self, enabled: bool):
        """Enable or disable interactive ROI selection."""
        self.is_selection_enabled = enabled
        if not enabled:
            self.is_dragging = False
            # We do NOT clear the ROI here, just stop editing.
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_roi_overlay(self, rect: QRect):
        """
        Set the ROI to display (in IMAGE coordinates).
        Pass an empty QRect() to clear.
        """
        self.roi_rect_img = rect
        self.update()

    def _get_image_draw_rect(self) -> QRect:
        """
        Calculate the rectangle where the image is actually drawn within the widget.
        """
        if not self.pixmap_frame or self.pixmap_frame.isNull():
            return QRect()

        target_rect = self.rect()
        scaled_size = self.pixmap_frame.scaled(
            target_rect.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ).size()

        x = (target_rect.width() - scaled_size.width()) // 2
        y = (target_rect.height() - scaled_size.height()) // 2

        return QRect(x, y, scaled_size.width(), scaled_size.height())

    def _widget_to_image_coords(self, widget_pt: QPoint) -> QPoint:
        """Convert widget coordinate to image coordinate."""
        draw_rect = self._get_image_draw_rect()
        if draw_rect.isEmpty():
            return QPoint(0, 0)

        # Relative to the drawn image
        rel_x = widget_pt.x() - draw_rect.x()
        rel_y = widget_pt.y() - draw_rect.y()

        # Scale factor
        scale_x = self.pixmap_frame.width() / draw_rect.width()
        scale_y = self.pixmap_frame.height() / draw_rect.height()

        img_x = int(rel_x * scale_x)
        img_y = int(rel_y * scale_y)

        # Clamp
        img_x = max(0, min(img_x, self.pixmap_frame.width()))
        img_y = max(0, min(img_y, self.pixmap_frame.height()))

        return QPoint(img_x, img_y)

    def _image_to_widget_rect(self, img_rect: QRect) -> QRect:
        """Convert image rect to widget rect for drawing."""
        draw_rect = self._get_image_draw_rect()
        if draw_rect.isEmpty() or img_rect.isEmpty():
            return QRect()

        scale_x = draw_rect.width() / self.pixmap_frame.width()
        scale_y = draw_rect.height() / self.pixmap_frame.height()

        x = int(img_rect.x() * scale_x) + draw_rect.x()
        y = int(img_rect.y() * scale_y) + draw_rect.y()
        w = int(img_rect.width() * scale_x)
        h = int(img_rect.height() * scale_y)

        return QRect(x, y, w, h)

    def mousePressEvent(self, event):
        if not self.is_selection_enabled or not self.pixmap_frame:
            return

        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.start_point = event.pos()
            self.current_point = event.pos()

            # Start a new selection, clear old visual temporarily until drag creates new one
            # Actually, standard behavior is drag starts new box
            self.update()

    def mouseMoveEvent(self, event):
        if not self.is_selection_enabled or not self.is_dragging:
            return

        self.current_point = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.is_selection_enabled or not self.is_dragging:
            return

        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.current_point = event.pos()

            # Finalize selection
            # Convert start/end to image coords
            p1 = self._widget_to_image_coords(self.start_point)
            p2 = self._widget_to_image_coords(self.current_point)

            # Create normalized rect (top-left, w, h)
            x = min(p1.x(), p2.x())
            y = min(p1.y(), p2.y())
            w = abs(p1.x() - p2.x())
            h = abs(p1.y() - p2.y())

            if w > 0 and h > 0:
                self.roi_rect_img = QRect(x, y, w, h)
                self.selection_changed.emit(self.roi_rect_img)

            self.update()

    def paintEvent(self, event):
        # 1. Draw Image
        if self.pixmap_frame and not self.pixmap_frame.isNull():
            painter = QPainter(self)

            draw_rect = self._get_image_draw_rect()
            painter.drawPixmap(draw_rect, self.pixmap_frame)

            # 2. Draw ROI (if exists)
            # Check if we are dragging currently
            rect_to_draw = QRect()

            if self.is_dragging:
                # While dragging, draw the transient rectangle from start_point to current_point
                # These are already in widget coords
                p1 = self.start_point
                p2 = self.current_point
                x = min(p1.x(), p2.x())
                y = min(p1.y(), p2.y())
                w = abs(p1.x() - p2.x())
                h = abs(p1.y() - p2.y())
                rect_to_draw = QRect(x, y, w, h)

            elif not self.roi_rect_img.isEmpty():
                # Not dragging, draw the committed/stored ROI
                rect_to_draw = self._image_to_widget_rect(self.roi_rect_img)

            if not rect_to_draw.isEmpty():
                # Draw the rect
                pen = QPen(QColor(0, 255, 0)) # Green
                pen.setWidth(2)
                painter.setPen(pen)
                # Fill semi-transparent
                painter.setBrush(QColor(0, 255, 0, 50))
                painter.drawRect(rect_to_draw)

        else:
            super().paintEvent(event)
