import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox

# Adjust path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera.real_camera import RealCamera
from camera.mock_camera import MockCamera
from gui.window import MainWindow

def main():
    # 1. Initialize Qt Application
    app = QApplication(sys.argv)

    # 2. Detect Camera
    # Check strict hardware availability unless forced to MOCK via env var (for development/sandbox)
    use_mock = os.environ.get("USE_MOCK_CAMERA", "0") == "1"

    camera = None

    if use_mock:
        print("Starting in MOCK mode (Environment Variable Set)")
        camera = MockCamera()
    else:
        # Check for real hardware
        if RealCamera.check_camera_availability():
            print("Real Camera Detected.")
            camera = RealCamera()
        else:
            # Constraint: "If camera NOT found: Show a clear error dialog. Allow app to close gracefully."
            # We create a dummy window just to show the message box if no main window exists yet,
            # or just show it and exit.

            # Since we haven't shown the main window yet, we can just pop the box.
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Camera Error")
            error_box.setText("No compatible camera found.")
            error_box.setInformativeText("Please connect an Arducam/Raspberry Pi Camera and try again.\n\n(Use 'rpicam-still --list-cameras' to verify)")
            error_box.setStandardButtons(QMessageBox.Ok)
            error_box.exec_()

            sys.exit(0)

    # 3. Launch Main Window
    window = MainWindow(camera)
    window.show()

    # 4. Run Event Loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
