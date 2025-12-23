from abc import ABC, abstractmethod
from src.core.types import RawImage

class ImageSource(ABC):
    """Abstract Base Class for Image Acquisition."""

    @abstractmethod
    def initialize(self) -> None:
        """Perform hardware startup or resource allocation."""
        pass

    @abstractmethod
    def capture(self) -> RawImage:
        """Acquire a single image."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        pass
