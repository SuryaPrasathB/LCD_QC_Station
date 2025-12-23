from abc import ABC, abstractmethod
import numpy as np

class CameraInterface(ABC):
    """
    Abstract base class for camera implementations.
    Ensures that RealCamera and MockCamera expose the same API.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the camera stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera stream and release resources."""
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        Retrieve the next frame from the camera.

        Returns:
            np.ndarray: The image frame in BGR format (compatible with OpenCV).
            None: If no frame is available or an error occurred.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the camera is currently running."""
        pass
