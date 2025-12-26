# Architectural Constraints & Context

## Hardware
- **Target:** Raspberry Pi 3B+
- **Camera:** Arducam 16MP / 64MP
- **Performance:** Limited CPU/RAM. Keep preview resolution to 640x480.

## Stack
- **OS:** Raspberry Pi OS (Bookworm)
- **GUI:** PyQt5 (QWidget based, no QML)
- **Camera Stack:** `Picamera2` (Python library). NO `rpicam-apps` or `libcamera-apps` via subprocess.
- **CV:** `opencv-python-headless` for frame handling.

## Requirements
- **Live View:** Use `Picamera2` dual stream (`lores` stream, YUV420) converted to BGR.
- **Capture:** Use `Picamera2` dual stream (`main` stream, RGB888) converted to BGR.
- **Architecture:** Modular. Camera logic must be decoupled from GUI.
- **Stability:** Must handle camera disconnects or missing hardware gracefully.
- **Framing:** Preview and Capture MUST have identical FOV. Use `ScalerCrop` to lock sensor geometry.

## Developer Modes
- **Mock Mode:** If hardware is missing (or forced via config), use `MockCamera` to generate synthetic frames for UI testing.
