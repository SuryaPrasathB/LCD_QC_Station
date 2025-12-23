# Architectural Constraints & Context

## Hardware
- **Target:** Raspberry Pi 3B
- **Camera:** Arducam 16MP / 64MP
- **Performance:** Limited CPU/RAM. Keep preview resolution to 640x480.

## Stack
- **OS:** Raspberry Pi OS (Bookworm/Bullseye)
- **GUI:** PyQt5 (QWidget based, no QML)
- **Camera Stack:** `rpicam-apps` (libcamera based). NO `picamera` or `picamera2`.
- **CV:** `opencv-python-headless` for frame handling.

## Requirements
- **Live View:** Use `rpicam-vid` (MJPEG to stdout) for the GUI preview stream.
- **Capture:** `rpicam-still` is reserved for high-res captures (not implemented in this step, but reserved).
- **Architecture:** Modular. Camera logic must be decoupled from GUI.
- **Stability:** Must handle camera disconnects or missing hardware gracefully.

## Developer Modes
- **Mock Mode:** If hardware is missing (or forced via config), use `MockCamera` to generate synthetic frames for UI testing.
