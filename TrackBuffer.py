
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BUFFER = ROOT / "buffer"

ROI_SIZE = 320
MODEL_IMGSZ = 416  # Match your TensorRT engine
MIN_COLOR_AREA = 100

COLOR_RANGES = {
    "red": [
        ((0, 80, 50), (10, 255, 255)),
        ((170, 80, 50), (179, 255, 255)),
    ],
    "blue": [
        ((90, 80, 50), (130, 255, 255)),
    ],
}


def find_color_rois(frame):
    """Find candidate centers using HSV color detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    centers = []

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    for ranges in COLOR_RANGES.values():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < MIN_COLOR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            centers.append((x + w // 2, y + h // 2))

    return centers


def make_roi(frame, cx, cy):
    """Produce a fixed-size ROI and its original-frame origin."""
    half = ROI_SIZE // 2
    x1, y1 = cx - half, cy - half

    height, width = frame.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2 = min(width, x1 + ROI_SIZE)
    sy2 = min(height, y1 + ROI_SIZE)

    roi = np.zeros(
        (ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8
    )

    if sx1 < sx2 and sy1 < sy2:
        dx, dy = sx1 - x1, sy1 - y1
        roi[
            dy:dy + (sy2 - sy1),
            dx:dx + (sx2 - sx1)
        ] = frame[sy1:sy2, sx1:sx2]

    return roi, x1, y1


def box_iou(a, b):
    """Intersection-over-union for duplicate detections."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    return intersection / max(
        area_a + area_b - intersection, 1e-9
    )


class Track:
    def __init__(self, model):
        self.model = model
        self.plate_xy = []
        self.avif_paths = []
        BUFFER.mkdir(parents=True, exist_ok=True)

        # Automatically exclude non-armor classes if named.
        self.armor_classes = {
            int(cls)
            for cls, name in model.names.items()
            if "armor" in name.lower()
            or "plate" in name.lower()
        }

    def process_frame(self, frame, frame_id):
        self.plate_xy.clear()
        files = []
        detections = []

        try:
            # STAGE 1: HSV color detection
            centers = find_color_rois(frame)

            # STAGE 2: Create ROI AVIF images
            for roi_id, (cx, cy) in enumerate(centers, 1):
                roi, x_offset, y_offset = make_roi(
                    frame, cx, cy
                )

                path = BUFFER / f"FRAME{frame_id}_{roi_id}.avif"

                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(
                    path, format="AVIF", quality=80
                )

                self.avif_paths.append(path)
                files.append((path, x_offset, y_offset))

            # STAGE 3: YOLO prediction for each ROI
            for path, ox, oy in files:
                result = self.model.predict(
                    source=str(path),
                    imgsz=MODEL_IMGSZ,
                    conf=0.1,
                    device=0,
                    save=False,
                    verbose=False,
                )[0]

                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls.item())

                    if (self.armor_classes
                            and cls not in self.armor_classes):
                        continue

                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().tolist()
                    )
                    confidence = float(box.conf.item())

                    # Transform ROI coordinates to frame coordinates
                    global_box = (
                        x1 + ox, y1 + oy,
                        x2 + ox, y2 + oy
                    )
                    detections.append((confidence, global_box))

            # STAGE 4: Remove overlapping duplicate detections
            accepted = []

            for confidence, box in sorted(
                detections, reverse=True
            ):
                if all(
                    box_iou(box, previous) < 0.5
                    for previous in accepted
                ):
                    accepted.append(box)

            self.plate_xy = [
                (
                    round((b[0] + b[2]) / 2),
                    round((b[1] + b[3]) / 2)
                )
                for b in accepted
            ]

            # Return a separate copy before clearing state
            return self.plate_xy.copy()

        finally:
            self.plate_xy.clear()

    def cleanup_avif_files(self):
        for path in self.avif_paths:
            path.unlink(missing_ok=True)
        self.avif_paths.clear()


if __name__ == "__main__":
    model_path = ROOT / (
        "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/"
        "weights/best.engine"
    )

    model = YOLO(str(model_path))
    tracker = Track(model)

    video_path = ROOT / "RoboMasterTestVideo2.mp4"
    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            coordinates = tracker.process_frame(
                frame, frame_id
            )

            print(f"FRAME {frame_id}: {coordinates}")
    finally:
        cap.release()
        tracker.cleanup_avif_files()
