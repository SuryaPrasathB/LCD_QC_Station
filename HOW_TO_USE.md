# How To Use: LCD QC Station

**Target Audience:** Operators, Process Engineers, and Maintenance Techs.
**Purpose:** Comprehensive guide to setting up, running, and maintaining the inspection station.

---

## 1. Prerequisites

Before you begin, ensure you have the following:

*   **Hardware:**
    *   Raspberry Pi 3B+ (or 4B).
    *   Compatible Camera (Pi Camera v2, v3, or HQ).
    *   Monitor, Mouse, and Keyboard connected to the Pi.
    *   Good, consistent lighting (Ring light recommended).
*   **OS:** Raspberry Pi OS (Bookworm recommended).
*   **Access:** Terminal access (or SSH) to the Raspberry Pi.

---

## 2. First-Time Setup (Raspberry Pi)

Follow these steps exactly to install the software on a fresh Raspberry Pi.

1.  **Open a Terminal** on the Raspberry Pi.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourOrg/LCD_QC_Station.git
    cd LCD_QC_Station
    ```
3.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `picamera2` and `PyQt5` are often best installed via system packages (`sudo apt install python3-picamera2 python3-pyqt5`), but the `requirements.txt` handles Python-level libraries.*

5.  **Verify Camera Availability:**
    Ensure no other app is using the camera.
    ```bash
    rpicam-still --list-cameras
    ```
    You should see "Available cameras" listed. If not, check your ribbon cable connection.

6.  **Verify Application:**
    Run the app in Mock mode first to check dependencies:
    ```bash
    export USE_MOCK_CAMERA=1
    python3 src/main.py
    ```
    If the window opens, the software is installed correctly. Close it and unset the variable (`unset USE_MOCK_CAMERA`) to use the real camera.

---

## 3. Understanding the Workflow (Conceptual)

*   **Reference Images ("Gold Standard"):**
    These are images of "perfect" products. The system compares every new product against these. If a product looks like a Reference, it passes.

*   **ROI (Region of Interest):**
    The camera sees a lot (the desk, the background). The ROI is a specific box you draw around the *actual part* you want to inspect (e.g., just the screen).

*   **PASS / FAIL:**
    *   **PASS:** The product is similar enough (score > threshold) to a Reference Image.
    *   **FAIL:** The product is too different.

*   **Overrides:**
    Sometimes the system makes a mistake.
    *   **False Fail:** System says FAIL, but it looks good. You mark it **PASS**.
    *   **False Pass:** System says PASS, but it has a defect. You mark it **FAIL**.
    *   *Important:* Overriding does **not** fix the system immediately. It saves the image so an engineer can "teach" the system later.

*   **Dataset Versions:**
    The system uses a "Version" (e.g., v1). When you teach the system new references, it creates "v2". This ensures the system doesn't change behavior unexpectedly.

---

## 4. Running the Application

1.  **Start the App:**
    ```bash
    source venv/bin/activate
    python3 src/main.py
    ```
2.  **The Interface:**
    *   **Live View:** Shows what the camera sees in real-time.
    *   **Capture Button:** Freezes the image to inspect it.
    *   **Status Panel:** Shows "PASS" (Green) or "FAIL" (Red).

---

## 5. Setting the ROI (VERY IMPORTANT)

You must set the Region of Interest (ROI) whenever the camera moves or the fixture changes.

1.  Place a **perfect sample** under the camera.
2.  Click **Capture**. The image freezes.
3.  Click **Clear ROI** (if one exists).
4.  **Click and Drag** on the image to draw a blue box around the product.
    *   *Tip:* Leave a small margin around the object, but exclude background clutter.
5.  Click **Set ROI**.
    *   The system saves this region.
    *   It also saves the current image as a **Reference Image**.

**⚠️ Warning:** Once set, **do not move the camera**. If the camera moves, the ROI will be looking at the wrong spot, and everything will FAIL.

---

## 6. Creating Reference Images

To add more "Gold Standards" (e.g., a variant of the product):

1.  Ensure **ROI** is already set.
2.  Place the new perfect sample.
3.  Click **Capture**.
4.  If the system allows (via "Commit Learning" workflow), this image can become a reference.
    *   *Currently:* The primary way to create a reference is during "Set ROI" or by manually adding files to `data/reference/vN/`.

---

## 7. Performing Inspection

1.  Place the test unit under the camera.
2.  Click **Capture**.
3.  Wait ~0.5 seconds.
4.  **Read the Result:**
    *   **Green PASS:** Unit is good.
    *   **Red FAIL:** Unit is bad.
    *   **Score:** A number (0.0 to 1.0). Higher is better (closer to reference).

5.  Click **Resume Live** to prepare for the next unit.

---

## 8. Operator Override Workflow

Use this when you disagree with the system.

**Scenario A: System says FAIL, but unit is Good.**
1.  Look at the screen. Result is **FAIL**.
2.  You inspect the part visually and confirm it is **Good**.
3.  Click **Mark as PASS**.
4.  The system logs this as a "Pending Learning" item.

**Scenario B: System says PASS, but unit is Bad.**
1.  Look at the screen. Result is **PASS**.
2.  You see a scratch.
3.  Click **Mark as FAIL**.
4.  The system logs this.

---

## 9. Dataset Versioning (CRITICAL)

When you have collected several "Pending Overrides" (e.g., good parts that failed), an Engineer should update the dataset.

1.  Click **Commit Learning**.
2.  The system takes all the "Mark as PASS" images and adds them to the Reference set.
3.  It creates a **New Version** (e.g., moves from `v1` to `v2`).
4.  It archives `v1` (so you can go back if needed).
5.  Future inspections will use `v2`.

---

## 10. Offline Model Training (PC Only)

**⚠️ NEVER run training on the Raspberry Pi.** It is too slow.

1.  **Copy Data:** Copy the `data/` folder from the Pi to your powerful PC.
2.  **Run Training Script:**
    ```bash
    # On PC
    python3 tools/train_embedding.py --data_dir ./data
    ```
3.  **Output:**
    *   This generates `models/embedding_v2.onnx`.
    *   It updates `models/model_meta.json`.

---

## 11. Deploying Model to Raspberry Pi

After training on PC:

1.  **Transfer Files:** Copy `models/embedding_v2.onnx` and `models/model_meta.json` back to the Pi.
    *   **Method 1 (Git Pull):**
        If you committed your models to the repo (caution: large files):
        ```bash
        cd LCD_QC_Station
        git pull
        ```
    *   **Method 2 (SCP):**
        ```bash
        scp models/* user@raspberrypi:~/LCD_QC_Station/models/
        ```
    *   **Method 3 (VS Code):**
        If using VS Code Remote-SSH, simply drag the files from your PC into the `models/` folder in the VS Code file explorer.

2.  **Restart App:**
    Restart `src/main.py` on the Pi to load the new model.

---

## 12. Running Inspection with ML

The system defaults to **Embedding (ML)** inspection.
*   If `models/embedding_v2.onnx` is missing, it falls back to **ORB (Legacy)**.
*   Check the logs or terminal output to verify: `Loading Embedding Model...`.

---

## 13. Developer Flags & Debugging

You can change behavior using Environment Variables:

*   **Force ORB Method:**
    ```bash
    export INSPECTION_METHOD=orb
    python3 src/main.py
    ```
*   **Force Embedding Method:**
    ```bash
    export INSPECTION_METHOD=embedding
    python3 src/main.py
    ```
*   **Mock Camera (No Hardware):**
    ```bash
    export USE_MOCK_CAMERA=1
    python3 src/main.py
    ```

---

## 14. Common Mistakes & Troubleshooting

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **"No compatible camera found"** | Cable loose or another app using camera. | Check cable. Run `killall python3`. |
| **Everything FAILS** | Camera moved. ROI is wrong. | Click **Resume**, then **Clear ROI**, then **Set ROI** again. |
| **Result is "Error"** | Missing reference images. | Set ROI (which creates a reference) or check `data/reference/`. |
| **System Falls back to ORB** | Model file missing. | Ensure `models/embedding_v2.onnx` exists. |
| **Result Mismatch** | Wrong dataset version. | Check `model_meta.json` matches active dataset version. |
| **System is slow (>1s)** | Running training? | Ensure you are NOT training on the Pi. |
| **Updates didn't apply** | App not restarted. | Restart the application after changing files. |

---

## 15. What This System Does NOT Do

*   **NO Online Learning:** It does not "get smarter" automatically as you work. You *must* click "Commit Learning" or run offline training.
*   **NO Auto-Threshold Tuning:** The system never changes the pass/fail sensitivity on its own.
*   **NO Auto-Centering:** If you put the part in the wrong place (outside the box), it will fail.
*   **NO Cloud Upload:** All images stay on the Pi (SD Card).
*   **NO Silent Updates:** The behavior is fixed until you explicitly change the Version.
