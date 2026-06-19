# Camera Sensor (Object Tracking)
# Karthikeyan Krishnan

# Core Python libraries
import cv2                    # OpenCV for image display and drawing
import threading              # Multithreading for camera, YOLO, and WebSocket
import time                   # Timing and delays
import uuid                   # Unique ID generation for sensor node
from datetime import datetime, timezone  # Timestamp generation (UTC)

from sensor import CameraThread, FrameBuffer, SensorClient

# To use YOLO machine learning models
from ultralytics import YOLO  # YOLO object detection model

YOLO_SIZE = 256       # Input resolution for YOLO inference
CONF_THRESH = 0.35    # Minimum confidence for detections
MODEL_FILE = "yolo26n.pt"  # YOLO model used

WINDOW = "Drone Detection & Tracking"  # OpenCV display window name

BUF_SIZE = 300        # Number of frames stored in circular buffer
CAMERA_CONFIG = {
    "driverType": "BASLER", # LOCAL or BASLER
    "width": 1920,         # Frame width
    "height": 1080,        # Frame height
    "fps": 30,             # Camera capture FPS
    "encodeFps": 15,       # Buffer FPS
    "ipAddress": ""
}

# Used to track objects moving in the video frame
# Assigns persistent IDs to detected objects using IoU matching
class Tracker:
    def __init__(self):
        self.next_id = 0      # Next available track ID
        self.objects = {}    # Active tracked objects

    # Update Tracks
    # Matches new detections to existing tracks
    def update(self, dets):
        updated = {}
        used = set()

        # Attempts to match existing objects to new detections
        for oid, obj in self.objects.items():
            box, cls, conf, ts = obj
            best_iou = 0
            best_det = None

            for index, det in enumerate(dets):
                if index in used:
                    continue
                iou = self.iou(box, det[0])
                if iou > best_iou:
                    best_iou = iou
                    best_det = index

            # Accept match if IoU is sufficient
            if best_iou > 0.3:
                updated[oid] = dets[best_det]
                used.add(best_det)

        # Create new IDs for unmatched detections
        for index, det in enumerate(dets):
            if index not in used:
                updated[self.next_id] = det
                self.next_id += 1

        self.objects = updated
        return self.objects

    # ---------------- Intersection over Union ----------------
    # Measures bounding box overlap
    @staticmethod
    def iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        area_of_intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_bbox_a = (a[2] - a[0]) * (a[3] - a[1])
        area_bbox_b = (b[2] - b[0]) * (b[3] - b[1])

        if area_bbox_a + area_bbox_b - area_of_intersection == 0:
            return 0

        return area_of_intersection / (area_bbox_a + area_bbox_b - area_of_intersection)

# Performs object detection, tracking, visualization, and event reporting
class YOLOWorker:
    def __init__(self, buf, client=None):
        self.buf = buf
        self.client = client
        self.model = YOLO(MODEL_FILE)
        self.tracker = Tracker()
        self.stop_flag = threading.Event()
        self.frame = None
        self.counter = 0

    # Starts inference thread
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    # Main inference loop
    def loop(self):
        while not self.stop_flag.is_set():
            img, ts = self.buf.get()
            if img is None:
                time.sleep(0.002)
                continue

            self.counter += 1

            # Run YOLO inference
            YOLO_results = self.model.predict(
                img,
                imgsz=YOLO_SIZE,
                conf=CONF_THRESH,
                device="cpu",
                verbose=False
            )[0]

            detections_list = []
            annotations_for_image = img.copy()

            # Extract the detections from the model
            if YOLO_results.boxes:
                for box in YOLO_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names[cls]
                    current_timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

                    detections_list.append([[x1, y1, x2, y2], name, conf, current_timestamp])

            # Update object tracks
            tracked = self.tracker.update(detections_list)

            # Draw annotations
            for tid, (bbox, name, conf, ts) in tracked.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotations_for_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotations_for_image, f"{name} ID:{tid} {conf:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            self.frame = annotations_for_image

            # Send detection event via WebSocket
            if self.client and self.client.is_connected() and tracked:
                event = {
                    "frame": self.counter,
                    "detections": [
                        {
                            "class": class_of_object,
                            "track_id": tid,
                            "bbox": bbox,
                            "confidence": conf_score
                        }
                        for tid, (bbox, class_of_object, conf_score, ts) in tracked.items()
                    ]
                }
                self.client.send_event(event)


# To initialize objects and to run the functions
def main():
    sensor_id = str(uuid.uuid4())

    buffer_from_main = FrameBuffer(BUF_SIZE)
    main_camera = CameraThread(buffer_from_main, sensor_id=sensor_id, fps=CAMERA_CONFIG["fps"])

    client = SensorClient(
        sensor_type="camera",
        sensor_id=sensor_id,
        location_relative=0,
        config=CAMERA_CONFIG
    )


    yolo = YOLOWorker(buffer_from_main, client=client)

    main_camera.start()
    yolo.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    try:
        while True:
            if yolo.frame is not None: # Shows Yolo output
                cv2.imshow(WINDOW, yolo.frame)
            elif main_camera.last_frame is not None: # Shows raw frame
                cv2.imshow(WINDOW, main_camera.last_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        main_camera.stop()
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
