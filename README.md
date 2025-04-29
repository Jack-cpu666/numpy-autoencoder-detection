# Pure Python/NumPy Unsupervised Object Detection Demo

This project demonstrates an end-to-end object detection pipeline built from scratch using only Python standard libraries, NumPy, and Pillow (for image I/O, drawing, and screen capture). It uses an unsupervised learning approach (Autoencoder) to detect recurring visual patterns learned from training images.

**Disclaimer:** This is an educational implementation focusing on demonstrating the core concepts (neural network basics, unsupervised learning, custom serialization, from-scratch pipeline) without relying on standard ML/CV frameworks. It is **not optimized for performance or detection accuracy** and will be significantly slower and less capable than libraries like PyTorch, TensorFlow, or OpenCV. The "detections" are low-level visual patterns, not semantic objects.

## Features

* **Neural Network from Scratch:** Autoencoder implemented using only NumPy for forward/backward passes and gradient descent.
* **Unsupervised Learning:** Learns to reconstruct common image patches from training data. No labels required.
* **Custom Model Format:** Uses a simple custom binary format (`.jack`) for model serialization.
* **Live Screen Detection:** Captures the screen (Windows implementation provided using `ctypes` via `Pillow.ImageGrab`) and applies the trained model using a sliding window.
* **Minimal Dependencies:** Only requires `numpy` and `Pillow`.

## Project Structure

├── data/
│   └── images/         # Place your training images here (JPEG/PNG)
├── train_model.py      # Script to train the autoencoder model
├── screen_detector.py  # Script to run live detection on the screen
├── model.jack          # Output of the training script (serialized model)
├── requirements.txt    # Python dependencies
└── README.md           # This file

## Setup

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Training Data:**
    * Create the `data/images/` directory.
    * Place several images (e.g., JPEGs, PNGs) containing various visual patterns into this directory. More diverse images representing common screen elements (icons, windows, text blocks, textures) might yield more interesting patterns. The autoencoder will learn to reconstruct common `16x16` grayscale patches from these images.

## How to Run

1.  **Train the Model:**
    * Run the training script from the project's root directory:
        ```bash
        python train_model.py
        ```
    * This will process images in `./data/images/`, train the autoencoder on extracted patches, and save the learned weights and architecture to `model.jack`.
    * Training can take some time depending on the number of images and the `EPOCHS` setting in `train_model.py`. The default is low (5 epochs) for a quick demonstration.

2.  **Run Live Screen Detection:**
    * Ensure the `model.jack` file exists (created by the training step).
    * Run the detection script:
        ```bash
        python screen_detector.py
        ```
    * The script will start capturing your screen.
    * It slides a `16x16` window across the screen, preprocesses each patch, and runs it through the trained autoencoder.
    * Patches that the autoencoder can reconstruct with low error (below `DETECTION_THRESHOLD` in `screen_detector.py`) are considered "detected patterns" and will be highlighted with a red bounding box.
    * Non-Maximum Suppression (NMS) is applied to reduce overlapping boxes.
    * An annotated version of your screen will be displayed (using Pillow's basic `show()` method, which might be slow or open multiple windows). FPS is shown in the top-left.
    * Press `Ctrl+C` in the terminal where the script is running to stop detection.

## Platform Notes (Screen Capture)

* The current `screen_detector.py` uses `Pillow.ImageGrab.grab()`. On Windows, this typically uses efficient `ctypes`-based screen capture.
* On **macOS** and **Linux**, `ImageGrab` might rely on external tools (like `scrot` on Linux) or different APIs. If it doesn't work out-of-the-box, you would need to:
    * Install necessary system dependencies (e.g., `scrot` on Linux: `sudo apt-get install scrot`).
    * OR: Replace the `capture_screen` function in `screen_detector.py` with a platform-specific implementation using `ctypes` to call native OS APIs (e.g., `CoreGraphics` on macOS, `Xlib` on Linux). This is non-trivial.

## Configuration & Limitations

* **Hyperparameters:** Key settings like `PATCH_SIZE`, `HIDDEN_DIM`, `LEARNING_RATE`, `EPOCHS`, `DETECTION_THRESHOLD`, `NMS_THRESHOLD`, `SCAN_STRIDE` are at the top of the respective Python scripts and can be tuned.
* **Performance:** Pure NumPy/Python implementation is very slow for image processing and neural network operations compared to optimized libraries. Expect low FPS during detection.
* **Detection Quality:** The autoencoder learns low-level patterns. It won't recognize complex objects like a cat or a specific application window unless trained extensively on very similar patches. Detections often correspond to common UI elements, textures, or edges present in the training data.
* **`.jack` Format:** The custom format is basic and not robust against corruption or version changes.
