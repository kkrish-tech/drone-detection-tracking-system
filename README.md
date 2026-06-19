# 🚁 Real-Time Drone Detection & Tracking System

A real-time AI-powered system that detects and tracks aerial objects using computer vision, edge AI acceleration, and automated camera/gimbal control. The project builds an end-to-end pipeline that connects live video capture, AI inference, object tracking, and real-time actuation for continuous aerial monitoring.

---

## 🧠 Technologies

- Python  
- OpenCV  
- YOLO (Ultralytics)  
- PyTorch  
- Hailo AI / Edge AI Accelerator  
- Basler pylon SDK (camera integration)  
- WebSockets  
- Serial Communication (gimbal control)  
- NumPy  
- Multithreading (Python threading)

---

## ⚙️ Features

### 🎯 Real-time Object Detection
Detects objects from live camera feed using a YOLO-based computer vision model.

### 🛰️ Drone Detection & Tracking
Identifies aerial objects and tracks them across frames for consistent monitoring.

### 🤖 Automated Camera / Gimbal Control
Adjusts camera pan and tilt in real time to keep detected targets centered in view.

### ⚡ Edge AI Acceleration
Runs inference on edge hardware (Hailo AI) for low-latency real-time performance.

### 📡 Live Event Streaming
Sends detection events in real time using WebSocket communication for external monitoring systems.

### 🧵 Multi-threaded Architecture
Separates camera capture, AI inference, and control logic for stable real-time performance.

---

## 🔧 The Process

The system is built as a continuous real-time feedback loop:

1. **Video Capture**
   - A Basler camera streams live video frames into a buffered pipeline.

2. **Frame Processing**
   - Frames are preprocessed and resized for model inference.

3. **AI Inference**
   - A YOLO-based model runs on each frame using edge AI acceleration for real-time detection.

4. **Detection Filtering**
   - Raw model outputs are filtered to extract valid object detections, focusing on aerial targets.

5. **Object Tracking**
   - Detected objects are tracked across frames to maintain identity and stability.

6. **Camera Actuation**
   - The system calculates object offset from frame center and sends commands to a gimbal to automatically adjust camera position.

7. **Live Communication**
   - Detection events and metadata are streamed to external systems using WebSockets.

---

## 📚 What I Learned

- How to design and build a **real-time AI system end-to-end**
- Integrating **computer vision with hardware control systems**
- Working with **edge AI acceleration for low-latency inference**
- Handling **real-time constraints like buffering, synchronization, and latency**
- Designing **multi-threaded pipelines for stable continuous processing**
- Building a **closed-loop system that connects perception → decision → action**

---

## 🚀 How It Can Be Improved

- Add advanced multi-object tracking (e.g., DeepSORT / ByteTrack)
- Improve detection accuracy for small or distant aerial objects
- Introduce GPU-based fallback inference for scalability
- Add trajectory prediction for moving targets
- Improve robustness in low-light and weather conditions
- Build a web dashboard for live monitoring and analytics
- Replace rule-based tracking with learning-based control strategies

---

## ▶️ Running the Project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Connect hardware
- Connect the Basler camera
- Ensure the gimbal serial connection is active
- Ensure the edge AI device (if used) is running

### 3. Run the system

python gimbal_scan_test.py
python CAM_DEMO.py

- Press q to exit the live view window



