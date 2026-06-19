#!/usr/bin/env python3

import serial
import sys
import time
import json
import signal
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
from pypylon import pylon

# ========================= SERIAL / GIMBAL CONFIG =========================
SERIAL_PORT = "/dev/ttyAMA0"
BAUDRATE = 9600
TIMEOUT = 0.5

PAN_MIN = -90
PAN_MAX = 90

SERVO_TILT_UPRIGHT = 30
SERVO_TILT_MAX_SAFE = 75

DISPLAY_UPRIGHT = 0
DISPLAY_MAX_TILT = 65

DEFAULT_PAN = 0
DEFAULT_SERVO_TILT = 65

PAN_STEP = 3
TILT_STEP = 3
PAN_SETTLE = 0.03
TILT_SETTLE = 0.06

CENTER_X_TOL = 0.12   # normalized tolerance around center
CENTER_Y_TOL = 0.12
TRACK_STEP = 2
LOCK_HOLD_FRAMES = 3
DETECTION_CONF = 0.25
# ========================================================================

# COCO‑class list (we'll treat "airplane" as drone)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)

# --- Colors for terminal output
TERM = {
    "RED":    "\033[31m",
    "GREEN":  "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE":   "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN":   "\033[36m",
    "BOLD":   "\033[1m",
    "RESET":  "\033[0m"
}

def servo_to_display(servo_angle):
    if servo_angle <= SERVO_TILT_UPRIGHT:
        return DISPLAY_UPRIGHT
    if servo_angle >= SERVO_TILT_MAX_SAFE:
        return DISPLAY_MAX_TILT
    ratio = (servo_angle - SERVO_TILT_UPRIGHT) / (SERVO_TILT_MAX_SAFE - SERVO_TILT_UPRIGHT)
    return round(DISPLAY_UPRIGHT + ratio * (DISPLAY_MAX_TILT - DISPLAY_UPRIGHT))

def send_command(ser, channel, angle):
    angle = int(angle)
    if channel == 'A':
        angle = max(PAN_MIN, min(PAN_MAX, angle))
        servo_angle = angle + 90
        display_angle = angle
    else:
        servo_angle = max(SERVO_TILT_UPRIGHT, min(SERVO_TILT_MAX_SAFE, angle))
        display_angle = servo_to_display(servo_angle)
    cmd = f"${channel}{servo_angle:03d}#"
    ser.write(cmd.encode("utf-8"))
    return display_angle, servo_angle

def move_to(ser, pan_angle, tilt_servo_angle):
    pan_disp, _ = send_command(ser, 'A', pan_angle)
    time.sleep(PAN_SETTLE)
    tilt_disp, _ = send_command(ser, 'B', tilt_servo_angle)
    time.sleep(TILT_SETTLE)
    return pan_disp, tilt_disp

def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))
    return x1, y1, x2, y2

def letterbox_image(img, target_size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(img, (new_w, new_h))
    pad_left = (target_size[0] - new_w) // 2
    pad_top = (target_size[1] - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_top, target_size[1] - new_h - pad_top,
        pad_left, target_size[0] - new_w - pad_left,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return padded, ratio, pad_left, pad_top

def get_detections_from_output(raw_output, ratio, pad_x, pad_y, w0, h0, conf_threshold):
    detections = []
    if isinstance(raw_output, list):
        for class_id, boxes in enumerate(raw_output):
            boxes = np.array(boxes)
            if boxes.size == 0:
                continue
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            for box in boxes:
                if len(box) < 5:
                    continue
                y_min, x_min, y_max, x_max, conf = map(float, box[:5])
                if conf < conf_threshold:
                    continue
                x1 = (x_min * 640 - pad_x) / ratio
                y1 = (y_min * 640 - pad_y) / ratio
                x2 = (x_max * 640 - pad_x) / ratio
                y2 = (y_max * 640 - pad_y) / ratio
                x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w0, h0)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((class_id, conf, x1, y1, x2, y2))
    return detections

def pick_drone_detections(detections, w, h):
    drones = []
    for class_id, conf, x1, y1, x2, y2 in detections:
        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class_{class_id}"
        if class_name == "airplane":
            drones.append(("drone", conf, x1, y1, x2, y2))
    return drones

def sweep_generator():
    pan = PAN_MIN
    tilt = SERVO_TILT_UPRIGHT
    direction = 1
    while True:
        yield pan, tilt
        if direction > 0:
            if pan + PAN_STEP <= PAN_MAX:
                pan += PAN_STEP
            else:
                if tilt + TILT_STEP <= SERVO_TILT_MAX_SAFE:
                    tilt += TILT_STEP
                    direction = -1
                else:
                    pan = PAN_MIN
                    tilt = SERVO_TILT_UPRIGHT
                    direction = 1
        else:
            if pan - PAN_STEP >= PAN_MIN:
                pan -= PAN_STEP
            else:
                if tilt + TILT_STEP <= SERVO_TILT_MAX_SAFE:
                    tilt += TILT_STEP
                    direction = 1
                else:
                    pan = PAN_MIN
                    tilt = SERVO_TILT_UPRIGHT
                    direction = 1

class FrameBuffer:
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)

    def get(self):
        with self.lock:
            return self.buffer.popleft() if self.buffer else None

def configure_camera_low_heat(camera):
    camera.Width.SetValue(1280)
    camera.Height.SetValue(720)
    camera.OffsetX.SetValue(0)
    camera.OffsetY.SetValue(0)
    camera.PixelFormat.SetValue("BayerRG8")
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(15.0)

def main():
    global running
    model_path = "yolov8n.hef"
    conf_threshold = DETECTION_CONF
    json_filename = f"sentry_drone_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detections_log = []

    print(f"{TERM['GREEN']}▶ Drone Detection Sentry System v1.0{TERM['RESET']}")
    print(f" Using {TERM['MAGENTA']}{SERIAL_PORT}{TERM['RESET']} @ {TERM['CYAN']}{BAUDRATE}{TERM['RESET']}")
    print(f" Scanning pan: {TERM['YELLOW']}{PAN_MIN} → +{PAN_MAX}{TERM['RESET']} | tilt: {TERM['YELLOW']}{DISPLAY_UPRIGHT} → {DISPLAY_MAX_TILT}{TERM['RESET']}")

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, 8, 'N', 1, timeout=TIMEOUT)
    time.sleep(0.5)
    print("✅ Serial port opened.")

    params = VDevice.create_params()
    try:
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    except:
        params.scheduling_algorithm = 1

    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(model_path)
        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)
        infer_model.set_batch_size(1)

        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()
            output_buffer = np.empty(infer_model.output().shape, dtype=np.float32)
            bindings.output().set_buffer(output_buffer)

            tl_factory = pylon.TlFactory.GetInstance()
            try:
                gige_tl = tl_factory.CreateTl("BaslerGigE")
                devices = gige_tl.EnumerateAllDevices()
            except Exception:
                devices = tl_factory.EnumerateDevices()

            if not devices:
                print(f"{TERM['RED']}❌ No Basler camera found!{TERM['RESET']}")
                return

            camera_device = None
            for dev in devices:
                if dev.GetSerialNumber() == "40724577":
                    camera_device = dev
                    break
            if camera_device is None:
                camera_device = devices[0]

            camera = pylon.InstantCamera(tl_factory.CreateDevice(camera_device))
            camera.Open()
            configure_camera_low_heat(camera)

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            camera.MaxNumBuffer = 20
            camera.OutputQueueSize = 20
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"📹 Camera streaming started (Serial {camera.DeviceSerialNumber.GetValue()})")

            frame_buffer = FrameBuffer(max_size=5)

            def camera_producer():
                while running and camera.IsGrabbing():
                    try:
                        grab_result = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
                        if grab_result.GrabSucceeded():
                            image = converter.Convert(grab_result)
                            frame_buffer.put(image.GetArray())
                        grab_result.Release()
                    except Exception:
                        time.sleep(0.001)

            threading.Thread(target=camera_producer, daemon=True).start()

            sweep = sweep_generator()
            hold_frames = 0
            last_move = time.time()
            frame_count = 0
            last_log_time = time.time()
            drone_frames = 0

            cv2.namedWindow("Sentry Drone Scan", cv2.WINDOW_AUTOSIZE)

            try:
                while running and camera.IsGrabbing():
                    frame = frame_buffer.get()
                    if frame is None:
                        time.sleep(0.001)
                        continue

                    frame_count += 1
                    frame_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    h0, w0 = frame.shape[:2]
                    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    letterboxed_rgb, ratio, pad_x, pad_y = letterbox_image(rgb_img)

                    input_tensor = np.expand_dims(letterboxed_rgb, axis=0).astype(np.uint8)
                    bindings.input().set_buffer(input_tensor)

                    configured_infer_model.run([bindings], timeout=1000)
                    raw_output = bindings.output().get_buffer()
                    detections = get_detections_from_output(raw_output, ratio, pad_x, pad_y, w0, h0, conf_threshold)
                    drones = pick_drone_detections(detections, w0, h0)

                    status = "SCANNING"
                    if drones:
                        status = f"{TERM['RED']}DRONE DETECTED{TERM['RESET']}"

                        # Draw red boxes and log to terminal
                        for i, (label, conf, x1, y1, x2, y2) in enumerate(drones):
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 0, 255), 2  # red
                            )
                            label_text = f"Drone {conf:.2f}"
                            cv2.putText(
                                frame, label_text,
                                (int(x1), max(30, int(y1) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2
                            )
                            print(f"[{frame_timestamp}] 🚁 {TERM['RED']}DRONE{TERM['RESET']} conf={conf:.2f} box=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

                        # Simple tracking: center on first drone
                        class_name, conf, x1, y1, x2, y2 = drones[0]
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        nx = cx / w0
                        ny = cy / h0

                        if abs(nx - 0.5) > CENTER_X_TOL or abs(ny - 0.5) > CENTER_Y_TOL:
                            pan_angle = int(round(max(PAN_MIN, min(PAN_MAX, (nx - 0.5) * 180.0))))
                            tilt_servo = int(round(max(SERVO_TILT_UPRIGHT, min(SERVO_TILT_MAX_SAFE, DEFAULT_SERVO_TILT + (ny - 0.5) * 30))))
                            send_command(ser, 'A', pan_angle)
                            time.sleep(PAN_SETTLE)
                            send_command(ser, 'B', tilt_servo)
                            time.sleep(TILT_SETTLE)

                        # Log to file
                        if conf >= DETECTION_CONF:
                            detections_log.append({
                                "frame_number": frame_count,
                                "timestamp": frame_timestamp,
                                "mode": "track",
                                "target": "drone_airplane",
                                "confidence": round(float(conf), 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            })
                            drone_frames += 1

                        hold_frames = LOCK_HOLD_FRAMES
                    else:
                        if hold_frames > 0:
                            hold_frames -= 1
                        else:
                            pan_angle, tilt_servo = next(sweep)
                            now = time.time()
                            if now - last_move > 0.01:
                                send_command(ser, 'A', pan_angle)
                                time.sleep(PAN_SETTLE)
                                send_command(ser, 'B', tilt_servo)
                                time.sleep(TILT_SETTLE)
                                last_move = now

                    # Add overlay UI text
                    cv2.putText(
                        frame, f"Status: {status.replace(TERM['RED'], '').replace(TERM['RESET'], '')}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"Detected drones: {drone_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, "Press 'q' to quit",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )

                    cv2.imshow("Sentry Drone Scan", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(f"{TERM['YELLOW']}⏹️  User quit requested.{TERM['RESET']}")
                        break

            finally:
                running = False
                print(f"🛑 Shutting down...")
                camera.StopGrabbing()
               
