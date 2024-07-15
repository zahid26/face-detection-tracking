# Real-Time Face Detection, Tracking and generate Embeddings.

This project implements real-time face detection and tracking using TensorFlow Lite for face detection and Deep SORT for tracking and generating embeddings.

## Features

- Real-time face detection using UltraLight models
- Face tracking using Deep SORT algorithm
- Embedding-based tracking for better accuracy
- Efficient and lightweight, suitable for embedded devices

## Requirements

- TensorFlow
- opencv-python
- numpy
- loguru
- deep-sort-realtime

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/zahid26/face-detection-tracking.git
    cd face-detection-tracking
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the TensorFlow Lite model for UltraLight face detection and place it in the project directory.

## Usage

To run the face detection and tracking:

```bash
python run.py --model_path path/to/your/model.tflite --camera_url camera_index/camera_url
```

## Refrences

- https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

- Deepsort
