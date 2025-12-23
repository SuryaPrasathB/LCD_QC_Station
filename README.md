# LCD/Generic Inspection System

## Overview
A robust, industrial-grade inspection system designed for Raspberry Pi 3B (Production) and Windows (Developer Mode). It performs visual quality checks on generic planar targets using reference-based computer vision.

## Features
*   **Dual Mode**: Production (Pi Camera) and Developer (Static Images).
*   **Robust Vision**: ORB-based alignment with fallback to translation for low-texture targets.
*   **Human-Assisted Learning**: Versioned datasets with "False Pass/Fail" override workflow.
*   **Interface**: Headless REST API (FastAPI) + Desktop GUI (PyQt6).
*   **Traceability**: Full SQLite logging and image retention policies.

## Setup

### Prerequisites
*   Python 3.9+
*   Raspberry Pi 3B (for Production) OR Windows (for Developer Mode)

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**Developer Mode (GUI)**
```bash
python main.py --mode developer --gui
```

**Production Mode (Headless)**
```bash
python main.py --mode production
```

## Architecture
See `design/DESIGN.md` for detailed architectural decisions.
