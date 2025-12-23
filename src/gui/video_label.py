from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPixmap, QColor

class VideoLabel(QLabel):
    """
    A custom QLabel that draws a scaled pixmap in its paintEvent.
    This prevents the 'resizing feedback loop' where setting a pixmap
    on a standard QLabel changes its sizeHint, forcing the window to grow,
    which triggers another resize/scale event.
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.pixmap_frame = None
        # Set size policy to Ignored so the layout determines the size,
        # not the content (pixmap) size.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; color: white; border: 1px solid #444;")

    def set_frame(self, pixmap: QPixmap):
        """
        Update the current frame and trigger a repaint.
        Does NOT call setPixmap() on the base class to avoid layout recalculations.
        """
        self.pixmap_frame = pixmap
        self.update() # Trigger paintEvent

    def paintEvent(self, event):
        """
        Custom paint event to draw the pixmap centered and scaled.
        """
        # If we have a frame, draw it
        if self.pixmap_frame and not self.pixmap_frame.isNull():
            painter = QPainter(self)

            # Calculate the rectangle to draw into (keeping aspect ratio)
            target_rect = self.rect()
            scaled_pixmap = self.pixmap_frame.scaled(
                target_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Center the image
            x = (target_rect.width() - scaled_pixmap.width()) // 2
            y = (target_rect.height() - scaled_pixmap.height()) // 2

            painter.drawPixmap(x, y, scaled_pixmap)

        else:
            # Fallback to standard QLabel drawing (for text)
            super().paintEvent(event)
