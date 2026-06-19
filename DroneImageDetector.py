# Image-Based Object Detection using YOLO
# Karthikeyan Krishnan

# Library Imports
import os                      # File and directory handling
import csv                     # CSV file writing
from datetime import datetime  # Timestamp generation
import cv2                     # OpenCV for image loading, drawing, and saving
from ultralytics import YOLO   # YOLO object detection framework

# Configuration variables
MODEL_NAME = "yolo11n.pt"  # YOLO model
CONF_THRESHOLD = 0.5       # Minimum confidence threshold for detections
IMGSZ = 640                # YOLO inference image size (pixels)

# Input/Output directories
INPUT_PATH = "images"          # Single image or directory of images
OUTPUT_PATH = "output_images"  # Directory for annotated images

# Timestamped CSV filename for detection logs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f"ImageDetections_{TIMESTAMP}.csv"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Image Object Detector Class
# Encapsulates YOLO model loading, inference, image annotation, and CSV logging.
class ImageObjectDetector:

    # Constructor
    # Loads the YOLO model into memory
    def __init__(self, model_name):
        self.model = YOLO(model_name)

    # Returns a list of image files from a single image path or multiple images in a directory
    def get_image_list(self, path):
        # If input is a single image file
        if os.path.isfile(path):
            return [path]

        # Otherwise, scan directory for image files
        return [
            os.path.join(path, file_extension)
            for file_extension in os.listdir(path)
            if file_extension.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]


    # Runs YOLO detection on all input images,
    # saves annotated images, and logs detections.
    def run(self, input_path):
        image_files = self.get_image_list(input_path)
        detections_csv = []  # Accumulates all detection records

        # Iterate through each image
        for img_index, image_path in enumerate(image_files, start=1):
            image = cv2.imread(image_path)

            # Skip unreadable images
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # ---------------------------
            # YOLO Inference
            # ---------------------------
            results = self.model.predict(
                image,
                conf=CONF_THRESHOLD,
                imgsz=IMGSZ,
                device="cpu",
                verbose=False
            )[0]

            # Copy image for annotation overlay
            annotated = image.copy()

            # Processes detection results
            if results.boxes is not None:
                for box in results.boxes:

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Extract confidence and class information
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    # Label text for overlay
                    label = f"{cls_name} {conf:.2f} [{x1},{y1},{x2},{y2}]"

                    # Draws Bounding Box on detected object
                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),  # Green box
                        2
                    )

                    # Draws Text Label (Outlined)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness_outline = 3
                    thickness_text = 1
                    text_origin = (x1, max(20, y1 - 10))

                    # Black outline for readability
                    cv2.putText(
                        annotated,
                        label,
                        text_origin,
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness_outline,
                        cv2.LINE_AA
                    )

                    # White text on top
                    cv2.putText(
                        annotated,
                        label,
                        text_origin,
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness_text,
                        cv2.LINE_AA
                    )

                    # Appends Detection to CSV Log
                    detections_csv.append([
                        os.path.basename(image_path),
                        img_index,
                        cls_name,
                        conf,
                        x1, y1, x2, y2
                    ])

            # Saves the annotated image
            output_file = os.path.join(
                OUTPUT_PATH,
                f"annotated_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_file, annotated)
            print(f"Saved: {output_file}")

        # Write all detections to CSV after processing completes
        self.write_csv(detections_csv)


    # Writes detection results to a timestamped CSV file
    def write_csv(self, rows):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Image",
                "ImageIndex",
                "Class",
                "Confidence",
                "x1",
                "y1",
                "x2",
                "y2"
            ])
            writer.writerows(rows)

        print(f"\n[CSV] Detection log saved to {CSV_FILE}")

# Initializes detector and starts processing
def main():
    detector = ImageObjectDetector(MODEL_NAME)
    detector.run(INPUT_PATH)

if __name__ == "__main__":
    main()
