# Face Dark Spots Detector (Live Camera)

This project uses OpenCV to detect faces in a live camera feed and then analyzes the skin region for darker spots (hyperpigmentation) using color space segmentation and contour detection.

## Prerequisites

1.  **Python 3.x**
2.  **Required Libraries:**
    ```bash
    pip install opencv-python numpy
    ```

## Project Setup

1.  **Create the Project Directory:**
    ```bash
    mkdir face_dark_spot_detector
    cd face_dark_spot_detector
    mkdir cascade_files
    ```

2.  **Download the Haar Cascade File:**
    Download `haarcascade_frontalface_default.xml` and place it in the `cascade_files/` directory.
    -   **Link:** https://github.com/opencv/opencv/raw/4.x/data/haarcascades/haarcascade_frontalface_default.xml

3.  **Create the Python Script:**
    Save the code above as `dark_spot_detector.py` in the root directory.

## How to Run

Execute the Python script from your terminal:

```bash
python dark_spot_detector.py