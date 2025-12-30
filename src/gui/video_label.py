from PyQt5.QtWidgets import QLabel, QSizePolicy, QInputDialog
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint, QSize
from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen
from typing import List, Dict, Optional

class VideoLabel(QLabel):
    """
    A custom QLabel that draws a scaled pixmap in its paintEvent.
    Also handles ROI selection (mouse dragging) and display.
    Supports Multiple ROIs.
    """
    # Emits the list of ROIs when changed
    # roi_list = [{'id': str, 'rect': QRect, 'status': str}]
    rois_changed = pyqtSignal(list)

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

        # List of ROIs. Each is a dict:
        # {'id': str, 'rect': QRect (image coords), 'status': 'pass'|'fail'|'none'}
        self.rois: List[Dict] = []

        # Counter for auto-ID
        self.roi_counter = 1

    def set_frame(self, pixmap: QPixmap):
        """Update the current frame and trigger a repaint."""
        self.pixmap_frame = pixmap
        self.update()

    def set_selection_enabled(self, enabled: bool):
        """Enable or disable interactive ROI selection."""
        self.is_selection_enabled = enabled
        if not enabled:
            self.is_dragging = False
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_rois(self, rois_data: List[Dict]):
        """
        Set the ROIs to display (in IMAGE coordinates).
        rois_data: List[Dict] usually containing {'rect', 'id', 'status'}
        """
        self.rois = rois_data
        # Reset counter based on existing IDs if they follow pattern?
        # For safety, let's max out.
        max_id = 0
        for r in self.rois:
            try:
                if r['id'].startswith('roi_'):
                    num = int(r['id'].replace('roi_', ''))
                    if num > max_id: max_id = num
            except:
                pass
        self.roi_counter = max_id + 1
        self.update()

    def clear_rois(self):
        self.rois = []
        self.roi_counter = 1
        self.update()
        self.rois_changed.emit(self.rois)

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
            p1 = self._widget_to_image_coords(self.start_point)
            p2 = self._widget_to_image_coords(self.current_point)

            x = min(p1.x(), p2.x())
            y = min(p1.y(), p2.y())
            w = abs(p1.x() - p2.x())
            h = abs(p1.y() - p2.y())

            if w > 10 and h > 10: # Minimum size filter
                new_rect = QRect(x, y, w, h)

                # Auto-generate ID
                new_id = f"roi_{self.roi_counter}"

                # Prompt for ID (Editable once)
                text, ok = QInputDialog.getText(self, "ROI ID", "Enter ID for this region:", text=new_id)
                if ok and text:
                    final_id = text
                    # If user accepted default, increment counter
                    if final_id == new_id:
                        self.roi_counter += 1
                else:
                    # User cancelled? Ignore ROI or use default?
                    # "Editable once" -> usually implies if cancelled, we discard or use default.
                    # Let's discard if cancelled to allow correcting mistakes.
                    self.update()
                    return

                self.rois.append({
                    'rect': new_rect,
                    'id': final_id,
                    'status': 'none'
                })
                self.rois_changed.emit(self.rois)

            self.update()

    def paintEvent(self, event):
        # 1. Draw Image
        if self.pixmap_frame and not self.pixmap_frame.isNull():
            painter = QPainter(self)
            # High quality drawing
            painter.setRenderHint(QPainter.Antialiasing)

            draw_rect = self._get_image_draw_rect()
            painter.drawPixmap(draw_rect, self.pixmap_frame)

            # 2. Draw Transient Drag Rect
            if self.is_dragging:
                p1 = self.start_point
                p2 = self.current_point
                x = min(p1.x(), p2.x())
                y = min(p1.y(), p2.y())
                w = abs(p1.x() - p2.x())
                h = abs(p1.y() - p2.y())
                rect_to_draw = QRect(x, y, w, h)

                pen = QPen(QColor(255, 255, 0)) # Yellow for creation
                pen.setWidth(2)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(rect_to_draw)

            # 3. Draw Stored ROIs
            for roi in self.rois:
                rect_img = roi['rect']
                status = roi.get('status', 'none')
                rid = roi.get('id', '')

                rect_widget = self._image_to_widget_rect(rect_img)
                if rect_widget.isEmpty():
                    continue

                # Color based on status
                if status == 'pass':
                    color = QColor(0, 255, 0) # Green
                elif status == 'fail':
                    color = QColor(255, 0, 0) # Red
                else:
                    color = QColor(0, 120, 255) # Blue/Neutral

                pen = QPen(color)
                pen.setWidth(3 if status == 'fail' else 2)
                painter.setPen(pen)

                # Fill transparent
                painter.setBrush(Qt.NoBrush)

                painter.drawRect(rect_widget)

                # Draw ID Label
                painter.setPen(QColor(255, 255, 255))
                painter.setBackgroundMode(Qt.OpaqueMode)
                painter.setBackground(QColor(0, 0, 0, 150))
                # Draw text just above or inside
                painter.drawText(rect_widget.topLeft() + QPoint(5, 15), rid)

        else:
            super().paintEvent(event)
