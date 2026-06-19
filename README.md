# YOLO-Driven Door State Monitoring Pipeline

## Overview

The YOLO-Driven Door State Monitoring Pipeline is a computer vision-based monitoring system designed to detect and track office entrance door states using CCTV footage.

The system utilizes a custom-trained YOLOv8 model to classify the door as either **Open** or **Closed** and employs a temporal state machine to reduce false detections and improve prediction stability. The application supports both live RTSP camera streams and recorded video files.

---

## Features

* Real-time door state detection using YOLOv8
* Support for RTSP CCTV streams
* Support for recorded video files
* Temporal anti-flutter state machine
* Automatic RTSP reconnection
* Door opening timestamp logging
* Door closing timestamp logging
* OpenCV-based monitoring dashboard
* Class-based implementation

---

## Dataset

The dataset was created from CCTV footage of an office entrance and manually annotated for door state classification.

### Classes

* `open`
* `close`

Due to privacy and security considerations, the complete dataset is not publicly available.

The dataset contains real-world variations including:

* Different door positions
* Lighting changes
* CCTV compression artifacts
* Minor viewpoint variations
* Office environment conditions

### Sample Dataset Images


Door Closed
<img width="353" height="616" alt="image" src="https://github.com/user-attachments/assets/1a4fbab2-743c-4872-b8b4-af9681b8816d" />


Door Open
<img width="325" height="630" alt="image" src="https://github.com/user-attachments/assets/9fd791bb-a853-405a-8766-c402370083fe" />


> Note: The original dataset has been excluded from this repository to protect organizational privacy and security.

---

## System Workflow

```text
Video Source (RTSP / Video File)
                │
                ▼
        Frame Acquisition
                │
                ▼
         YOLOv8 Inference
                │
                ▼
     Door State Classification
        (Open / Closed)
                │
                ▼
      Temporal State Machine
                │
                ▼
      State Transition Logic
                │
                ▼
        Event Timestamping
                │
                ▼
      Real-Time Visualization
```

---

## Project Structure

```text
YOLO-Driven Door State Monitoring Pipeline/
│
├── data_preparation/
│   └── data_fetch.py
│
│
├── model/
│   ├── best.pt
│   └── result/
│       ├── confusion_matrix.png
│       ├── results.csv
│       └── results.png
│
├── src/
│   ├── main.py
│   └── train.py
│
├── requirements.txt
│
└── README.md
```

---

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLOv8
* PyTorch
* FFmpeg
* RTSP Streaming

---


## Requirements:
All required dependencies and package versions are listed in the `requirements.txt` file included in this repository.

Install dependencies:

```bash
pip install -r requirements.txt
```



## Model Training

The model is trained using the lightweight YOLOv8 Nano architecture.

### Training Configuration

| Parameter  | Value     |
| ---------- | --------- |
| Model      | YOLOv8n   |
| Epochs     | 20        |
| Image Size | 640 × 640 |
| Device     | CPU       |

> Note: The values shown above are example training settings used during development. Users may adjust the number of epochs, image size, batch size, device selection, and other training parameters according to their dataset size, computational resources, and performance requirements.
### Start Training

```bash
python src/train.py
```

The training pipeline loads the pretrained YOLOv8n model and fine-tunes it using the custom door-state dataset.

Training results are generated inside:

```text
model/result/
```

---

## Running the Monitoring System

### RTSP Mode

Configure:

```python
monitor = OfficeDoorMonitor(source_type="rtsp")
```

Run:

```bash
python src/main.py
```

### Video File Mode

Configure:

```python
monitor = OfficeDoorMonitor(source_type="video")
```

Run:

```bash
python src/main.py
```

---

## Anti-Flutter State Machine

To prevent false state transitions caused by temporary prediction errors, the system requires multiple consecutive detections before accepting a state change.

```text
Closed
   │
   ▼
Pending Open
   │
   ▼
N Consecutive Open Detections
   │
   ▼
Open
```
> Note:The consecutive detection threshold (N) is configurable. During testing, a threshold of 5 consecutive detections was used as an example. Users may increase or decrease this value depending on camera quality, frame rate, environmental conditions, and desired sensitivity. Higher values provide greater stability, while lower values allow faster state transitions.

This temporal smoothing mechanism improves robustness in real-world CCTV environments.

---

## Output Dashboard

The monitoring dashboard displays:

* Current Door Status
* Last Door Opened Timestamp
* Last Door Closed Timestamp
* Live Video Feed

Example:

```text
STATUS: OPEN

LAST OPENED:
2026-06-18 10:35:42

LAST CLOSED:
2026-06-18 10:20:15
```

---

## Future Improvements

* Event database logging
* Email notifications
* Multi-camera monitoring
* Web dashboard integration

---

## Author

**Nandana MT**

This project is released under a free and open-source license for academic and non-commercial use. Users are free to study, modify, and reuse the code with proper attribution.
