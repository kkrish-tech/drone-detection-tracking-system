# Camera Sensor (Object Tracking)
# Karthikeyan Krishnan

# Core Python libraries
import cv2                    # OpenCV for image display and drawing
import threading              # Multithreading for camera, YOLO, and WebSocket
import time                   # Timing and delays
import json                   # JSON serialization for WebSocket messages
import uuid                   # Unique ID generation for sensor node
from datetime import datetime, timezone  # Timestamp generation (UTC)

# Camera SDK (Basler)
from pypylon import pylon     # Interface for Basler industrial cameras

# To use YOLO machine learning models
from ultralytics import YOLO  # YOLO object detection model

# To use WebSockets
from websocket import WebSocketApp  # WebSocket client for event streaming

# Camera and processing parameters
BUF_SIZE = 300        # Number of frames stored in circular buffer
FPS = 60              # Camera frame rate limit
YOLO_SIZE = 256       # Input resolution for YOLO inference
CONF_THRESH = 0.35    # Minimum confidence for detections
MODEL_FILE = "yolo11n.pt"  # YOLO model used

WINDOW = "Drone Detection & Tracking"  # OpenCV display window name

# Handles persistent WebSocket connection to a node/server
# Used to transmit detection events in near-real time
class WSClient:
    MAX_RETRIES = 3    # Maximum reconnection attempts

    def __init__(self, url, node_id, node_type, location):
        # Connection metadata
        self.url = url
        self.id = node_id
        self.type = node_type
        self.loc = location

        # Connection state
        self.connected = False
        self.status = "DISCONNECTED"
        self.ws = None
        self.retries = 0

        # Starts background connection thread
        self.start()

    # Attempts to maintain a live WebSocket connection
    # Loops until maximum attempts has been reached
    def start(self):
        def loop():
            while self.retries < self.MAX_RETRIES:
                self.ws = WebSocketApp(
                    self.url,
                    on_open=self.on_open,
                    on_message=self.on_msg,
                    on_close=self.on_close,
                    on_error=self.on_err
                )

                # Websocket attempts to connect to the node
                # Sends a blocking call; returns on disconnect
                self.ws.run_forever(ping_interval=10, ping_timeout=5)

                # Retries the websocket connection if connection fails
                if not self.connected:
                    self.retries += 1
                    print(f"[WebSocket] Attempt {self.retries}/{self.MAX_RETRIES} failed. Retrying in 3s...")
                    time.sleep(3)

            # Stops after attempting to connect 3 times to the node
            self.update_status("FAILED")
            print("[WebSocket] Max retries reached. Connection aborted.")
        threading.Thread(target=loop, daemon=True).start()

    # ---- Callback methods for the WebSocket Client ----
    # To connect to the node by sending a handshake message
    def on_open(self, ws):
        self.connected = True
        self.retries = 0

        # Handshake message identifying this sensor
        handshake = {
            "id": self.id,
            "type": self.type,
            "locationRelative": self.loc
        }

        ws.send(json.dumps(handshake))
        self.update_status("CONNECTED")
        print("[WebSocket] Connected. Handshake sent.")

    # To display messages that are being received
    def on_msg(self, msg):
        print(f"[WebSocket] Message received: {msg}")

    # To close a WebSocket connection
    def on_close(self, ws, code, reason):
        self.connected = False

    # Called on connection error
    def on_err(self, ws, err):
        self.connected = False

    # Sends JSON-encoded event data to the connected Node
    def send(self, data):
        if self.ws and self.connected:
            self.ws.send(json.dumps(data))

    # Tracks and prints connection state changes
    def update_status(self, new_status):
        if self.status != new_status:
            self.status = new_status
            print(f"[WebSocket Status] {self.status}")


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


# Circular Frame Buffer used to store camera frames which would be used by the YOLO model
class FrameBuffer:
    def __init__(self, size):
        self.size = size
        self.frames = [None] * size
        self.ts = [None] * size
        self.write_index = 0      # Write index
        self.read_index = 0      # Read index
        self.frame_count = 0    # Frame count
        self.lock = threading.Lock()

    # Adds a frame to the buffer with the timestamp at which it occurred
    def put(self, frame, timestamp):
        with self.lock:
            self.frames[self.write_index] = frame
            self.ts[self.write_index] = timestamp
            self.write_index = (self.write_index + 1) % self.size

            if self.frame_count < self.size:
                self.frame_count += 1
            else:
                self.read_index = (self.read_index + 1) % self.size

    # Retrieves the oldest frame
    def get(self):
        with self.lock:
            if self.frame_count == 0:
                return None, None

            frame = self.frames[self.read_index]
            ts = self.ts[self.read_index]
            self.read_index = (self.read_index + 1) % self.size
            self.frame_count -= 1
            return frame, ts


# Main Camera Thread
# Continuously captures frames from Basler camera
class CameraThread:
    def __init__(self, buf, fps=20, buf_size=300):
        self.buf = buf
        self.fps = fps
        self.max_buf = buf_size
        self.stop_flag = threading.Event()

        # Discovers and opens the first available Basler camera
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera detected!")

        self.cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        self.cam.Open()

        # Configures the pixel format for the camera
        # Uses BGR8 to read color pixels
        try:
            self.cam.PixelFormat.Value = "BGR8"
        except:
            self.cam.PixelFormat.SetValue("BGR8")

        # Image converter
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed

        # Camera settings
        self.cam.MaxNumBuffer = self.max_buf
        self.cam.TriggerMode.SetValue("Off")
        self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(self.fps)

        self.cam.StartGrabbing(pylon.GrabStrategy_OneByOne)

    # Starts capture thread
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    # Stops capture and releases camera
    def stop(self):
        self.stop_flag.set()
        if self.cam.IsGrabbing():
            self.cam.StopGrabbing()
        self.cam.Close()

    # Capture loop
    def loop(self):
        while not self.stop_flag.is_set():
            grab = self.cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                frame = self.conv.Convert(grab).GetArray()
                ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                self.buf.put(frame, ts)
            grab.Release()


# Performs object detection, tracking, visualization, and event reporting
class YOLOWorker:
    def __init__(self, buf, ws=None):
        self.buf = buf
        self.ws = ws
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
            if self.ws and self.ws.connected and tracked:
                event = {
                    "sensorId": self.ws.id,
                    "eventLocation": self.ws.loc,
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
                self.ws.send(event)


# To initialize objects and to run the functions
def main():

    buffer_from_main = FrameBuffer(BUF_SIZE)
    main_camera = CameraThread(buffer_from_main)

    main_WebSocket = WSClient(
        url="ws://127.0.0.1:5050/ws/sensors",
        node_id=str(uuid.uuid4()),
        node_type="camera",
        location=0
    )

    yolo = YOLOWorker(buffer_from_main, ws=main_WebSocket)

    main_camera.start()
    yolo.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    try:
        while True:
            if yolo.frame is not None:
                cv2.imshow(WINDOW, yolo.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        main_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
