import cv2
import time
import json
import threading
from datetime import datetime, timezone
from ultralytics import YOLO

# Try importing Basler support
try:
    from pypylon import pylon
except ImportError:
    pylon = None

# ===================== CONFIG =====================

CAMERA_TYPE = "BASLER"  # "LOCAL" or "BASLER"
CAMERA_INDEX = 0
WIDTH = 1920
HEIGHT = 1080
TARGET_FPS = 15
YOLO_MODEL = "best.pt"
CONF_THRESH = 0.15

WINDOW = "YOLO Detection Feed"
FRAME_INTERVAL = 1.0 / TARGET_FPS

FOCAL_LENGTH_PX = 780
CLASS_WIDTHS_M = {
    "person": 0.45,
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.6,
    "motorcycle": 0.8,
    "bicycle": 0.6,
    "drone": 0.35,
    "airplane": 30.0,
    "bird": 0.4,
    "default": 0.5,
}

SMOOTH_ALPHA = 0.3

# ===================== TIMESTAMPED OUTPUT =====================

RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT_JSON = f"detections_{RUN_TS}.json"

# ===================== FRAME BUFFER =====================

class FrameBuffer:
    def __init__(self, size: int = 5):
        self.size = size
        self.frames = [None] * size
        self.ts = [None] * size
        self.write_index = 0
        self.read_index = 0
        self.frame_count = 0
        self.lock = threading.Lock()

    def put(self, frame, timestamp):
        with self.lock:
            self.frames[self.write_index] = frame
            self.ts[self.write_index] = timestamp
            self.write_index = (self.write_index + 1) % self.size
            if self.frame_count < self.size:
                self.frame_count += 1
            else:
                self.read_index = (self.read_index + 1) % self.size

    def get(self):
        with self.lock:
            if self.frame_count == 0:
                return None, None
            frame = self.frames[self.read_index]
            ts = self.ts[self.read_index]
            self.read_index = (self.read_index + 1) % self.size
            self.frame_count -= 1
            return frame, ts

# ===================== CAMERA THREAD =====================

class CameraThread:
    def __init__(self, buf: FrameBuffer):
        self.buf = buf
        self.stop_flag = threading.Event()
        self.cap = None
        self.basler_cam = None

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.stop_flag.set()
        if self.cap:
            self.cap.release()
        if self.basler_cam:
            try:
                self.basler_cam.StopGrabbing()
            except Exception:
                pass
            try:
                self.basler_cam.Close()
            except Exception:
                pass

    def loop(self):
        # Initialize camera
        if CAMERA_TYPE.upper() == "LOCAL":
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            if not self.cap.isOpened():
                print("ERROR: Failed to open local camera")
                return

        elif CAMERA_TYPE.upper() == "BASLER":
            if not pylon:
                print("ERROR: pypylon not installed")
                return
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                print("ERROR: No Basler cameras found")
                return
            camera_device = tl_factory.CreateDevice(devices[0])
            self.basler_cam = pylon.InstantCamera(camera_device)
            self.basler_cam.Open()
            self.basler_cam.Width.Value = WIDTH
            self.basler_cam.Height.Value = HEIGHT
            self.basler_cam.PixelFormat.Value = "BGR8"  #Change to NV12 later
            self.basler_cam.AcquisitionFrameRateEnable.SetValue(True)
            self.basler_cam.AcquisitionFrameRate.SetValue(TARGET_FPS)
            self.basler_cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        else:
            print("ERROR: Invalid CAMERA_TYPE")
            return

        while not self.stop_flag.is_set():
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            frame = None

            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.001)
                    continue

            elif self.basler_cam:
                try:
                    if self.basler_cam.IsGrabbing():
                        grab_result = self.basler_cam.RetrieveResult(
                            5000, pylon.TimeoutHandling_ThrowException
                        )
                        if grab_result.GrabSucceeded():
                            frame = grab_result.Array
                        grab_result.Release()
                    else:
                        time.sleep(0.001)
                        continue
                except Exception:
                    # Camera stopped or program exiting
                    break

            if frame is not None:
                self.buf.put(frame, ts)
            time.sleep(0.001)

# ===================== UTILS =====================

def draw_text(img, text, pos, font_scale=0.6, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def print_table(detections, frame_counter):
    if not detections:
        return
    headers = ["ID", "Class", "Confidence", "Distance", "BBox", "Timestamp"]
    col_widths = [5, 15, 14, 12, 28, 30]
    print("\n" + "-" * sum(col_widths))
    print(f"Frame {frame_counter} Detections:")
    print("".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("-" * sum(col_widths))
    for det in detections:
        print(
            str(det["ID"]).ljust(col_widths[0]) +
            det["Class"].ljust(col_widths[1]) +
            f"{det['Confidence']:.2f}%".ljust(col_widths[2]) +
            f"{det['Distance_m']:.2f}m".ljust(col_widths[3]) +
            str(det["BBox"]).ljust(col_widths[4]) +
            det["Timestamp"].ljust(col_widths[5])
        )
    print("-" * sum(col_widths))

# ===================== MAIN =====================

def main():
    buf = FrameBuffer(size=10)
    cam_thread = CameraThread(buf)
    cam_thread.start()

    model = YOLO(YOLO_MODEL)
    prev_distances = {}
    frames_data = []
    frames_data_lock = threading.Lock()
    frame_counter = 0
    fps = 0.0
    prev_time = time.perf_counter()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, WIDTH, HEIGHT)

    try:
        while True:
            frame, ts = buf.get()
            if frame is None:
                time.sleep(0.001)
                continue

            frame_counter += 1

            results = model.predict(frame, imgsz=640, conf=CONF_THRESH, device="cpu", verbose=False)
            detections = []

            for box_data in results[0].boxes if results and results[0].boxes else []:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                cls_id = int(box_data.cls[0])
                conf = float(box_data.conf[0])
                name = model.names[cls_id]

                real_width = CLASS_WIDTHS_M.get(name, CLASS_WIDTHS_M["default"])
                pixel_width = max(1, x2 - x1)
                raw_distance = (real_width * FOCAL_LENGTH_PX) / pixel_width

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                key = (name, cx // 10, cy // 10)
                distance_m = SMOOTH_ALPHA * raw_distance + \
                             (1 - SMOOTH_ALPHA) * prev_distances.get(key, raw_distance)
                prev_distances[key] = distance_m

                label = f"{name} {conf:.2f} {distance_m:.1f}m"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                draw_text(frame, label, (x1, max(20, y1 - 5)))

                detections.append({
                    "ID": len(detections),
                    "Class": name,
                    "Confidence": round(conf * 100, 2),
                    "Distance_m": round(distance_m, 2),
                    "BBox": [x1, y1, x2, y2],
                    "Timestamp": ts
                })

            with frames_data_lock:
                frames_data.append({
                    "Frame": frame_counter,
                    "Timestamp": ts,
                    "Detections": detections
                })

            # FPS calculation
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / (now - prev_time)
            prev_time = now

            hud = f"Resolution: {frame.shape[1]}x{frame.shape[0]} | FPS: {fps:.1f}"
            draw_text(frame, hud, (10, 30), font_scale=0.8)

            print_table(detections, frame_counter)
            cv2.imshow(WINDOW, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Sleep to maintain target FPS
            elapsed = time.perf_counter() - now
            if FRAME_INTERVAL - elapsed > 0:
                time.sleep(FRAME_INTERVAL - elapsed)

    finally:
        cam_thread.stop()
        cv2.destroyAllWindows()

        with frames_data_lock:
            output = {
                "RunTimestampUTC": RUN_TS,
                "CameraType": CAMERA_TYPE,
                "Resolution": f"{WIDTH}x{HEIGHT}",
                "TargetFPS": TARGET_FPS,
                "Frames": frames_data
            }

        with open(OUTPUT_JSON, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved detection log to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
