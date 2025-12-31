from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QByteArray

def bytes_to_pixmap(data: bytes) -> QPixmap:
    """Converts bytes (e.g. JPEG) to QPixmap."""
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap
