
from pathlib import Path
import time
import threading

from time import perf_counter
from collections import deque, defaultdict

from ultralytics import YOLO
import numpy as np
import cv2
import torch

from concurrent.futures import ThreadPoolExecutor


# ============================================================
# CONFIGURATION
# ============================================================

root = Path(__file__).resolve().parent

videos = [
    "[BLUR_TEST]RoboMasterTestVideo3.mp4",
    "RoboMasterTestVideo2.mp4",
]

skip = 3

# Profiling
ENABLE_PROFILING = True
PROFILE_WINDOW = 300
REPORT_INTERVAL = 300

# Asynchronous processing
MAX_PENDING = 2
WORKER_COUNT = 1

# YOLO
ROI_SIZE = 96
CONFIDENCE = 0.1
YOLO_DEVICE = 0

# Maximum number of crops submitted in one inference batch.
# Must be compatible with the TensorRT engine's batch profile.
YOLO_BATCH_SIZE = 4

# Detection output
PRINT_DETECTIONS = True


# ============================================================
# PERFORMANCE MONITOR
# ============================================================

class PerformanceMonitor:

    def __init__(self, window=300):

        self.timings = defaultdict(
            lambda: deque(maxlen=window)
        )

        self.counts = defaultdict(
            lambda: deque(maxlen=window)
        )

        self.lock = threading.Lock()

    def record(self, name, elapsed):

        if not ENABLE_PROFILING:
            return

        with self.lock:
            self.timings[name].append(elapsed * 1000)

    def record_count(self, name, value):

        if not ENABLE_PROFILING:
            return

        with self.lock:
            self.counts[name].append(value)

    def report(self, title="PERFORMANCE REPORT"):

        print("\n")
        print("=" * 85)
        print(title)
        print("=" * 85)

        print(
            f"{'METRIC':<28}"
            f"{'AVG':>12}"
            f"{'P50':>12}"
            f"{'P95':>12}"
            f"{'MAX':>12}"
        )

        print("-" * 85)

        with self.lock:

            timing_data = {
                name: list(values)
                for name, values in self.timings.items()
                if values
            }

            count_data = {
                name: list(values)
                for name, values in self.counts.items()
                if values
            }

        for name, values in timing_data.items():

            values = np.asarray(values)

            print(
                f"{name:<28}"
                f"{np.mean(values):>12.2f}"
                f"{np.percentile(values, 50):>12.2f}"
                f"{np.percentile(values, 95):>12.2f}"
                f"{np.max(values):>12.2f}"
            )

        print("\nCOUNTERS")
        print("-" * 85)

        for name, values in count_data.items():

            values = np.asarray(values)

            print(
                f"{name:<28}"
                f"AVG: {np.mean(values):>10.2f}"
                f"MAX: {np.max(values):>10.0f}"
            )

        print("=" * 85)


# ============================================================
# GPU SYNCHRONIZATION
# ============================================================

def synchronize_gpu():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# OBJECT CLASSES
# ============================================================

class plate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class robot:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    color: str
    plates: plate


class Track:

    def __init__(self, x1, y1, x2, y2, margin=80):

        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2

        self.width = x2 - x1
        self.height = y2 - y1

        self.margin = margin
        self.missing_frames = 0

    def update(self, x1, y1, x2, y2):

        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2

        self.width = x2 - x1
        self.height = y2 - y1

        self.missing_frames = 0

    def get_roi(self, frame_width, frame_height):

        return (
            max(0, int(self.x - self.width / 2 - self.margin)),
            max(0, int(self.y - self.height / 2 - self.margin)),
            min(frame_width, int(self.x + self.width / 2 + self.margin)),
            min(frame_height, int(self.y + self.height / 2 + self.margin))
        )


tracks = []


# ============================================================
# COLOR DETECTION
# ============================================================

def color_detection(frame, timings=None):

    start = perf_counter()

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    if timings is not None:
        timings["hsv_conversion"] = perf_counter() - start

    color_ranges = {

        "red": [
            (
                np.array([0, 100, 100]),
                np.array([10, 255, 255])
            ),
            (
                np.array([170, 100, 100]),
                np.array([180, 255, 255])
            )
        ],

        "blue": [
            (
                np.array([100, 100, 100]),
                np.array([130, 255, 255])
            )
        ]
    }

    detections = []

    start = perf_counter()

    for color, ranges in color_ranges.items():

        mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8
        )

        for lower, upper in ranges:

            mask |= cv2.inRange(
                hsv,
                lower,
                upper
            )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(contour) > 100:

                x, y, w, h = cv2.boundingRect(contour)

                detections.append(
                    (color, (x, y, w, h))
                )

    if timings is not None:
        timings["mask_and_contours"] = perf_counter() - start

    return detections


# ============================================================
# BATCHED ROI INFERENCE
# ============================================================

def predict_rois(
    model,
    frame,
    roi_size=32,
    submitted_at=None
):

    timings = {}

    worker_start = perf_counter()

    # --------------------------------------------------------
    # QUEUE LATENCY
    # --------------------------------------------------------

    if submitted_at is not None:

        timings["queue_latency"] = (
            perf_counter() - submitted_at
        )

    H, W = frame.shape[:2]

    # --------------------------------------------------------
    # COLOR DETECTION
    # --------------------------------------------------------

    color_start = perf_counter()

    color_boxes = color_detection(
        frame,
        timings
    )

    timings["color_detection"] = (
        perf_counter() - color_start
    )

    timings["color_regions"] = len(color_boxes)

    # --------------------------------------------------------
    # GENERATE ROI CROPS
    # --------------------------------------------------------

    roi_start = perf_counter()

    crops = []
    roi_offsets = []

    for color, (x, y, w, h) in color_boxes:

        cx = x + w // 2
        cy = y + h // 2

        x1 = max(
            0,
            min(cx - roi_size // 2, W - roi_size)
        )

        y1 = max(
            0,
            min(cy - roi_size // 2, H - roi_size)
        )

        x2 = min(W, x1 + roi_size)
        y2 = min(H, y1 + roi_size)

        crop = frame[y1:y2, x1:x2]

        # Ensure every crop has the expected dimensions.
        # This matters for batching and edge-of-frame ROIs.
        if crop.shape[:2] != (roi_size, roi_size):

            crop = cv2.copyMakeBorder(
                crop,
                0,
                roi_size - crop.shape[0],
                0,
                roi_size - crop.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        crops.append(crop)
        roi_offsets.append((x1, y1))

    timings["roi_generation"] = (
        perf_counter() - roi_start
    )

    timings["roi_count"] = len(crops)

    # --------------------------------------------------------
    # BATCHED YOLO INFERENCE
    # --------------------------------------------------------

    plate_boxes = []

    yolo_total = 0.0

    for batch_start in range(0, len(crops), YOLO_BATCH_SIZE):

        batch_crops = crops[
            batch_start:batch_start + YOLO_BATCH_SIZE
        ]

        batch_offsets = roi_offsets[
            batch_start:batch_start + YOLO_BATCH_SIZE
        ]

        synchronize_gpu()

        yolo_start = perf_counter()

        results = model.predict(
            source=batch_crops,
            imgsz=roi_size,
            conf=CONFIDENCE,
            device=YOLO_DEVICE,
            batch=len(batch_crops),
            save=False,
            verbose=False
        )

        synchronize_gpu()

        yolo_total += perf_counter() - yolo_start

        # ----------------------------------------------------
        # COORDINATE CONVERSION
        # ----------------------------------------------------

        coordinate_start = perf_counter()

        for result, (x1, y1) in zip(
            results,
            batch_offsets
        ):

            boxes = result.boxes.xyxy.cpu().numpy()

            for bx1, by1, bx2, by2 in boxes:

                plate_boxes.append([
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1
                ])

        timings["coordinate_conversion"] = (
            timings.get("coordinate_conversion", 0)
            + perf_counter() - coordinate_start
        )

    timings["yolo_inference"] = yolo_total

    timings["detected_plates"] = len(plate_boxes)

    # --------------------------------------------------------
    # TOTAL WORKER EXECUTION
    # --------------------------------------------------------

    timings["worker_total"] = (
        perf_counter() - worker_start
    )

    return (
        np.asarray(
            plate_boxes,
            dtype=np.float32
        ).reshape(-1, 4),

        timings
    )


# ============================================================
# VIDEO TRACKING
# ============================================================

def track_video(
    model,
    video_path,
    enemyColor,
    allyColor,
    maxPixDistance
):

    video_path = Path(video_path)

    if not video_path.is_absolute():
        video_path = root / video_path

    frame_count = 0

    start_time = perf_counter()

    monitor = PerformanceMonitor(
        window=PROFILE_WINDOW
    )

    executor = ThreadPoolExecutor(
        max_workers=WORKER_COUNT
    )

    pending = deque()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        executor.shutdown(wait=True)

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    print("\n")
    print("=" * 85)
    print(f"PROCESSING: {video_path.name}")
    print("=" * 85)

    # --------------------------------------------------------
    # FUTURE RETRIEVAL
    # --------------------------------------------------------

    def retrieve_result(frame_id, future):

        wait_start = perf_counter()

        plate_boxes, timings = future.result()

        monitor.record(
            "future_wait",
            perf_counter() - wait_start
        )

        # Timing measurements
        for name, elapsed in timings.items():

            if name in (
                "color_regions",
                "roi_count",
                "detected_plates"
            ):
                continue

            monitor.record(
                name,
                elapsed
            )

        # Count measurements
        monitor.record_count(
            "color_regions",
            timings["color_regions"]
        )

        monitor.record_count(
            "roi_count",
            timings["roi_count"]
        )

        monitor.record_count(
            "detected_plates",
            timings["detected_plates"]
        )

        if PRINT_DETECTIONS:

            for x1, y1, x2, y2 in plate_boxes:

                print(
                    f"Frame: {frame_id} | "
                    f"Box: ({x1:.0f}, {y1:.0f}, "
                    f"{x2:.0f}, {y2:.0f})"
                )

        return plate_boxes

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    try:

        while True:

            # FRAME READING
            read_start = perf_counter()

            success, frame = cap.read()

            monitor.record(
                "frame_read",
                perf_counter() - read_start
            )

            if not success:
                break

            frame_count += 1

            # ASYNCHRONOUS SUBMISSION
            submit_start = perf_counter()

            submitted_at = perf_counter()

            future = executor.submit(
                predict_rois,
                model,
                frame,
                ROI_SIZE,
                submitted_at
            )

            monitor.record(
                "task_submission",
                perf_counter() - submit_start
            )

            pending.append(
                (frame_count, future)
            )

            # BOUND QUEUE SIZE
            if len(pending) > MAX_PENDING:

                frame_id, completed = pending.popleft()

                retrieve_result(
                    frame_id,
                    completed
                )

            # PERIODIC REPORT
            if frame_count % REPORT_INTERVAL == 0:

                elapsed = perf_counter() - start_time

                print("\n")
                print(
                    f"Progress: {frame_count} frames | "
                    f"Elapsed: {elapsed:.2f}s | "
                    f"Throughput: {frame_count / elapsed:.2f} FPS"
                )

                monitor.report(
                    f"ROLLING REPORT - FRAME {frame_count}"
                )

        # ----------------------------------------------------
        # COLLECT REMAINING FUTURES
        # ----------------------------------------------------

        while pending:

            frame_id, completed = pending.popleft()

            retrieve_result(
                frame_id,
                completed
            )

    finally:

        cap.release()

        executor.shutdown(
            wait=True
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    elapsed = perf_counter() - start_time

    print("\n")
    print("=" * 85)
    print(f"VIDEO COMPLETE: {video_path.name}")
    print("=" * 85)

    print(f"Total frames: {frame_count}")
    print(f"Total elapsed: {elapsed:.2f}s")

    if elapsed > 0:

        print(
            f"Overall throughput: "
            f"{frame_count / elapsed:.2f} FPS"
        )

    monitor.report(
        "FINAL PERFORMANCE REPORT"
    )

    return frame_count


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = (
        root
        / "runs/detect/[LARGE]Robomasters.v6i.yolo26_EPOCH100/weights/best.engine"
    )

    print("Loading YOLO model...")

    model = YOLO(
        str(model_path)
    )

    print("Model loaded.")

    # Team color configuration
    enemyColor = "red"
    allyColor = "blue"

    maxPixDistance = 80

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    # Warmup with a dummy frame to reduce first-inference
    # initialization effects in the actual benchmark.
    #
    # This is not included in the video frame count.

    print("Warming up model...")

    dummy_frame = np.zeros(
        (ROI_SIZE, ROI_SIZE, 3),
        dtype=np.uint8
    )

    try:

        model.predict(
            source=[dummy_frame],
            imgsz=ROI_SIZE,
            conf=CONFIDENCE,
            device=YOLO_DEVICE,
            batch=1,
            save=False,
            verbose=False
        )

        synchronize_gpu()

    except Exception as error:

        print(
            "Warmup warning:",
            error
        )

    print("Warmup complete.")

    # --------------------------------------------------------
    # PROCESS VIDEOS
    # --------------------------------------------------------

    for video in videos:

        track_video(
            model,
            video,
            enemyColor,
            allyColor,
            maxPixDistance
        )


if __name__ == "__main__":

    main()
