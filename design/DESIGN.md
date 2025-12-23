# LCD/Generic Inspection System - Design Document

## 1. Introduction
This document outlines the architecture for a robust, industrial-grade inspection system designed for Raspberry Pi 3B (Production) and Windows (Developer Mode). The system performs visual quality checks on generic planar targets (LCDs, printed text, etc.) using a reference-based approach with human-assisted learning.

## 2. System Architecture

### 2.1 High-Level Components
*   **Hardware Abstraction Layer (HAL)**: Normalizes input from Camera or Disk.
*   **Core Logic**: Stateless Vision Engine + Stateful Inspection Controller.
*   **Data Layer**: SQLite (Metadata) + File System (Blobs).
*   **Presentation**: REST API (Headless) + PyQt6 (GUI).

### 2.2 Directory Structure
```text
/
├── design/                 # Design docs
├── data/                   # Runtime storage (GitIgnored)
│   ├── images/             # Raw storage (Year/Month/Day/ID.jpg)
│   ├── datasets/           # Versioned Reference sets
│   ├── db/                 # production.sqlite
│   └── logs/               # system.log
├── src/
│   ├── core/
│   │   ├── vision/         # Alignment, Diff, Features
│   │   ├── engine.py       # Orchestrator
│   │   ├── dataset.py      # Versioning & Management
│   │   └── types.py        # Pydantic Models
│   ├── hal/
│   │   ├── camera/         # rpicam-still wrapper
│   │   └── files/          # Developer input
│   ├── infra/
│   │   ├── database.py     # SQLite/ORM
│   │   ├── storage.py      # Retention logic
│   │   └── logging.py      # Structured logger
│   ├── api/                # FastAPI
│   └── gui/                # PyQt6
└── tests/
```

## 3. Core Modules

### 3.1 Vision Engine (`src/core/vision`)
**Strategy**: Reference-Based Statistical Alignment.

#### Alignment Pipeline
1.  **Feature Extraction**: Extract ORB features from Input and Reference.
2.  **Homography**: Compute transformation matrix.
3.  **Validation (Safety Mechanism)**:
    *   Check number of keypoints > `MIN_KEYPOINTS` (e.g., 10).
    *   Check homography inlier ratio.
    *   **Fallback**: If validation fails (low texture/glare), fallback to **Translation-Only Alignment** (Phase Correlation or Template Matching within ROI).
4.  **Warp**: Apply transformation (or translation) to align Input to Reference.

#### Comparison & Decision
1.  **Diff**: `absdiff(WarpedInput, Reference)`.
2.  **Clean**: GaussianBlur + Thresholding -> Binary Diff Map.
3.  **Metric Calculation**:
    *   Extract blobs (contours) from Diff Map.
    *   **Robustness**: Cap max contribution of any single blob (prevents glare spikes).
    *   Compute `Score = 1.0 - (Sum(CappedBlobs) / ToleranceFactor)`.
4.  **Result**: PASS if Score > `CONFIDENCE_THRESHOLD`.

### 3.2 Dataset & Learning (`src/core/dataset.py`)
**Philosophy**: No blind updates. Changes are staged and committed.

#### Terminology
*   **False FAIL**: System said FAIL, User says PASS.
    *   *Action*: Add image to "Allowed Variants" (or update statistical mean) in next version.
*   **False PASS**: System said PASS, User says FAIL.
    *   *Action*: Tighten tolerance thresholds or mask out noise regions.

#### Workflow
1.  **Override**: User marks result as incorrect via API/GUI. Stored in `PendingOverrides` table.
2.  **Review**: (Optional) User reviews pending overrides.
3.  **Commit**: User triggers "Commit Version".
    *   System generates `Dataset vN+1`.
    *   Pending overrides are processed into the new model state.
    *   Old version `vN` is preserved for rollback.

### 3.3 Hardware Abstraction (`src/hal`)
*   **`ImageSource`**: Abstract Base Class.
*   **`PiCameraSource`**:
    *   Uses `subprocess` to call `rpicam-still`.
    *   Manages startup latency and timeouts.
    *   Returns `RawImage` object.
*   **`FileImageSource`**:
    *   Watches/Reads from a local directory.
    *   Simulates capture delay (optional) to match Pi timing in dev.

## 4. Data Persistence

### 4.1 Database (SQLite)
*   **`inspections`**: `id`, `timestamp`, `result`, `score`, `file_path`, `dataset_version_id`.
*   **`datasets`**: `id`, `version_tag`, `created_at`, `parameters_json`.
*   **`overrides`**: `id`, `inspection_id`, `original_result`, `new_result`, `status`.

### 4.2 File System
*   **Images**: Stored by date: `data/images/YYYY/MM/DD/<UUID>.jpg`.
*   **Retention**:
    *   Keep ALL `FAIL`.
    *   Keep ALL `OVERRIDDEN`.
    *   Rotate `PASS` (FIFO based on disk usage or count).

## 5. Interface

### 5.1 REST API (FastAPI)
*   `POST /inspect`: Trigger inspection.
*   `GET /results/last`: Get latest result.
*   `POST /results/{id}/override`: Submit human correction.
*   `POST /dataset/commit`: Finalize learning.

### 5.2 GUI (PyQt6)
*   **Modes**: Operator (Simple Pass/Fail), Developer (Detailed view, ROI setup, Override).
*   **Developer Features**:
    *   Load folder of test images.
    *   Step through images.
    *   Visualize Diff Map and Keypoints.

## 6. Constraints & Tradeoffs
*   **Latency**: Target < 2.5s. Deep learning (CNNs) avoided in favor of classical CV for speed on Pi 3B.
*   **Lighting**: Assumes controlled or semi-controlled lighting. Large variations may require multiple Reference images (Multi-modal reference).
