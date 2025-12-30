# LCD QC Station

## 1. Project Overview

**LCD_QC_Station** is a specialized visual quality inspection system designed for industrial environments. It performs automated pass/fail verification of planar surfaces (such as LCD screens, printed labels, or circuit boards) by comparing a live camera feed against a set of validated "Gold Standard" reference images.

### Target Use Case
*   **Industrial Inspection:** Final quality control on assembly lines.
*   **Visual Validation:** Checking for defects, alignment issues, or missing components.

### Core Philosophy
*   **Offline Learning:** No learning happens on the device during production. Models are trained offline to ensure stability.
*   **Frozen Inference:** The decision logic is deterministic and versioned.
*   **Traceability:** Every inspection and operator override is logged with strict dataset versioning.

## 2. System Architecture (High Level)

The system is built on a modular "Hardware Abstraction Layer" architecture to support both the target hardware (Raspberry Pi) and developer environments (PC).

*   **Camera & ROI:** Uses `Picamera2` for hardware control. Strict 4:3 aspect ratio is enforced to guarantee identical Field of View (FOV) between low-resolution previews and high-resolution captures.
*   **Inspection Pipeline:**
    *   **Primary:** Deep Learning Embedding (MobileNetV2 via ONNX). Calculates Cosine Distance between the captured Region of Interest (ROI) and reference embeddings.
    *   **Fallback:** ORB (Oriented FAST and Rotated BRIEF) feature matching for environments where ML inference is not suitable.
*   **Offline vs Online:**
    *   **Online (Pi):** Inference only. No model weight updates.
    *   **Offline (PC):** Training and Dataset Management.
*   **Dataset Versioning:** Reference images and overrides are managed in immutable versioned directories (e.g., `v1`, `v2`) to prevent silent drift in quality standards.

## 3. Features

*   **Live Preview:** Low-latency 640x480 video feed for positioning.
*   **ROI Selection:** Interactive GUI tool to define the exact region to inspect.
*   **Capture & Inspect:** One-click process to capture a high-res (8MP) image and perform analysis.
*   **PASS / FAIL Decision:** Automated decision based on a configurable similarity threshold.
*   **Operator Override:** Operators can flag False Positives or False Negatives. These overrides are saved for future offline training but do **not** change immediate behavior.
*   **Dataset Versioning:** "Commit Learning" feature to package validated references and overrides into a new immutable dataset version.
*   **Offline-Trained ML Inference:** Lightweight CNN (MobileNetV2) running on CPU.
*   **Deterministic Behavior:** Given the same input and dataset version, the result is always identical.

## 4. Directory Structure

*   **`src/`**: Application source code.
    *   `src/camera/`: Hardware abstraction for `Picamera2` and Mock cameras.
    *   `src/core/`: Core logic for inspection, embedding extraction, and dataset management.
    *   `src/gui/`: PyQt5 user interface.
*   **`data/`**: Runtime data storage.
    *   `data/reference/`: Versioned reference images (`v1/`, `v2/`).
    *   `data/overrides/`: Logs of operator overrides.
    *   `data/logs/`: Inspection history.
*   **`models/`**: Frozen machine learning models (`embedding_v2.onnx`) and metadata.
*   **`tools/`**: Offline utility scripts for model training (`train_embedding.py`).
*   **`tests/`**: Unit and integration tests.

## 5. Inspection Pipeline Summary

1.  **Camera Capture:** High-resolution image (3280x2464) is captured.
2.  **ROI Extraction:** The defined Region of Interest is cropped from the capture.
3.  **Preprocessing:** Image is resized (128x128) and normalized.
4.  **Embedding:** MobileNetV2 extracts a 128-dimensional vector representation.
5.  **Distance Calculation:** Cosine distance is computed against all active Reference Embeddings.
6.  **Decision:**
    *   If `min(distance) <= threshold`: **PASS**.
    *   Otherwise: **FAIL**.

**Note:** The system explicitly does **not** perform online learning. An operator override only saves the data; it does not update the decision boundary until a new model/dataset is explicitly deployed.

## 6. Hardware & OS Requirements

*   **Hardware:** Raspberry Pi 3B+ (or newer).
*   **Camera:** Raspberry Pi Camera Module v2, v3, or High Quality Camera (Arducam compatible).
*   **Display:** HDMI monitor (minimum 1024x768).
*   **OS:** Raspberry Pi OS (Bookworm or Bullseye) with `libcamera` support enabled.

## 7. Software Dependencies (High Level)

*   **Python:** 3.9+
*   **Camera:** `picamera2` (pre-installed on RPi OS).
*   **GUI:** `PyQt5`.
*   **CV:** `opencv-python-headless` (or `opencv-contrib-python`).
*   **ML Runtime:** `onnxruntime` (or `tflite-runtime`).
*   **Math:** `numpy`, `scikit-image`.

## 8. Safety & Design Guarantees

*   **No Silent Behavior Change:** The inspection logic never changes automatically. It only changes when an engineer explicitly commits a new dataset version.
*   **Deterministic Outputs:** The same image will always produce the same score.
*   **Offline-Only Learning:** Prevents "poisoning" the model with bad data during production.
*   **Full Traceability:** Every inspection result links back to a specific Dataset Version and Reference Image ID.

## 9. Project Status

*   **Step 1-2:** Core setup and architecture (Completed)
*   **Step 3:** Basic Camera & GUI (Completed)
*   **Step 4:** ROI Selection (Completed)
*   **Step 5:** Inspection Logic & Persistence (Completed)
*   **Step 6:** Offline Learning Tools (Completed)
*   **Step 7:** Model Integration (Completed)
*   **Step 8:** Refinement & Verification (Completed)
*   **Step 9:** Deployment Packaging (**Not Implemented**)
